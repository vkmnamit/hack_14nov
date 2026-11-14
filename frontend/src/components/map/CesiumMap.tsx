import { useEffect, useRef } from 'react';
import {
  Cartesian3,
  Cartesian2,
  Cartographic,
  Cesium3DTileset,
  BoxEmitter,
  Color,
  ColorGeometryInstanceAttribute,
  GeometryInstance,
  ImageryLayer,
  Matrix4,
  Ion,
  IonImageryProvider,
  Math as CesiumMath,
  PerInstanceColorAppearance,
  ParticleSystem,
  PointPrimitive,
  PointPrimitiveCollection,
  Primitive,
  Rectangle,
  ScreenSpaceEventHandler,
  ScreenSpaceEventType,
  UrlTemplateImageryProvider,
  Viewer as CesiumViewer,
  CesiumTerrainProvider,
  createOsmBuildingsAsync,
} from 'cesium';

import type { FloodFrame, TerrainGrid } from '@/lib/demoData';
import { buildGridGeometry } from '@/lib/cesiumGridGeometry';

interface CesiumMapProps {
  terrain: TerrainGrid | null;
  frame: FloodFrame | null;
  longitude: number;
  latitude: number;
  cameraHeight: number;
  ionToken: string;
  cameraPreset: CameraPreset;
  rainfallMmHr: number | null;
  rainEnabled: boolean;
  flowEnabled: boolean;
  flowSpeedMultiplier: number;
  windDirection: number;
  windSpeed: number;
  radarEnabled: boolean;
  radarOpacity: number;
  radarUrl: string;
  radarTimes: number[];
  onPick?: (info: { lon: number; lat: number; elevation: number; depth: number } | null) => void;
}

export type CameraPreset = 'global' | 'india' | 'bengaluru' | 'study';

/** Half-extent of the simulated tile in degrees — roughly a 1.6 km urban tile. */
const HALF_EXTENT_DEG = 0.0075;

const BANDS: { max: number; color: string; alpha: number }[] = [
  { max: 0.05, color: '#9BD8F5', alpha: 0.6 },
  { max: 0.15, color: '#00D4FF', alpha: 0.7 },
  { max: 0.3, color: '#1560D4', alpha: 0.75 },
  { max: 0.5, color: '#FFB547', alpha: 0.8 },
  { max: Infinity, color: '#FF4545', alpha: 0.88 },
];

function bandIndexForDepth(depth: number): number {
  for (let i = 0; i < BANDS.length; i++) if (depth < BANDS[i].max) return i;
  return BANDS.length - 1;
}

function rectFor(longitude: number, latitude: number): Rectangle {
  return Rectangle.fromDegrees(
    longitude - HALF_EXTENT_DEG,
    latitude - HALF_EXTENT_DEG,
    longitude + HALF_EXTENT_DEG,
    latitude + HALF_EXTENT_DEG
  );
}

function safeRemove(viewer: CesiumViewer, primitive: Primitive | null): Primitive | null {
  if (primitive && !primitive.isDestroyed()) {
    viewer.scene.primitives.remove(primitive);
  }
  return null;
}

function createRainTexture(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = 8;
  canvas.height = 32;
  const context = canvas.getContext('2d');
  if (context) {
    const gradient = context.createLinearGradient(0, 0, 0, 32);
    gradient.addColorStop(0, 'rgba(190, 235, 255, 0)');
    gradient.addColorStop(0.35, 'rgba(190, 235, 255, 0.8)');
    gradient.addColorStop(1, 'rgba(255, 255, 255, 0.95)');
    context.fillStyle = gradient;
    context.fillRect(3, 0, 2, 32);
  }
  return canvas;
}

interface FlowParticle {
  point: PointPrimitive;
  lon: number;
  lat: number;
  velocityX: number;
  velocityY: number;
  elevation: number;
  depth: number;
  originLon: number;
  originLat: number;
}

function flyToPreset(
  viewer: CesiumViewer,
  preset: CameraPreset,
  longitude: number,
  latitude: number,
  cameraHeight: number
): void {
  const targets: Record<CameraPreset, { longitude: number; latitude: number; height: number; pitch: number }> = {
    global: { longitude: 78, latitude: 20, height: 19_000_000, pitch: -90 },
    india: { longitude: 78, latitude: 22, height: 4_500_000, pitch: -70 },
    bengaluru: { longitude: 77.5946, latitude: 12.9716, height: 85_000, pitch: -52 },
    study: { longitude, latitude: latitude - 0.011, height: cameraHeight, pitch: -34 },
  };
  const target = targets[preset];
  viewer.camera.flyTo({
    destination: Cartesian3.fromDegrees(target.longitude, target.latitude, target.height),
    orientation: {
      heading: CesiumMath.toRadians(preset === 'global' ? 0 : 12),
      pitch: CesiumMath.toRadians(target.pitch),
      roll: 0,
    },
    duration: preset === 'global' ? 2.2 : 1.6,
  });
}

export default function CesiumMap({
  terrain,
  frame,
  longitude,
  latitude,
  cameraHeight,
  ionToken,
  cameraPreset,
  rainfallMmHr,
  rainEnabled,
  flowEnabled,
  flowSpeedMultiplier,
  windDirection,
  windSpeed,
  radarEnabled,
  radarOpacity,
  radarUrl,
  radarTimes,
  onPick,
}: CesiumMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const viewerRef = useRef<CesiumViewer | null>(null);
  const terrainPrimitiveRef = useRef<Primitive | null>(null);
  const floodPrimitiveRef = useRef<Primitive | null>(null);
  const buildingsRef = useRef<Cesium3DTileset | null>(null);
  const imageryLayerRef = useRef<ImageryLayer | null>(null);
  const radarLayerRef = useRef<ImageryLayer | null>(null);
  const rainSystemRef = useRef<ParticleSystem | null>(null);
  const flowPointsRef = useRef<PointPrimitiveCollection | null>(null);
  const flowParticlesRef = useRef<FlowParticle[]>([]);
  const flowEnabledRef = useRef(flowEnabled);
  const flowSpeedMultiplierRef = useRef(flowSpeedMultiplier);
  const flowBoundsRef = useRef({ longitude, latitude });

  useEffect(() => {
    flowEnabledRef.current = flowEnabled;
    flowSpeedMultiplierRef.current = flowSpeedMultiplier;
    flowBoundsRef.current = { longitude, latitude };
  }, [flowEnabled, flowSpeedMultiplier, longitude, latitude]);

  // --- One-time viewer bootstrapping --------------------------------------
  useEffect(() => {
    if (!containerRef.current) return;

    const viewer = new CesiumViewer(containerRef.current, {
      animation: false,
      baseLayerPicker: false,
      fullscreenButton: false,
      geocoder: false,
      homeButton: false,
      infoBox: false,
      sceneModePicker: false,
      selectionIndicator: false,
      timeline: false,
      navigationHelpButton: false,
      baseLayer: new ImageryLayer(
        new UrlTemplateImageryProvider({
          url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          credit: '© OpenStreetMap contributors',
        })
      ),
    });

    viewer.scene.backgroundColor = Color.fromCssColorString('#030405');
    viewer.scene.globe.baseColor = Color.fromCssColorString('#080f16');
    viewer.scene.globe.enableLighting = true;
    viewer.scene.globe.showGroundAtmosphere = true;
    viewer.scene.fog.enabled = true;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = true;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;
    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.shadows = true;

    viewerRef.current = viewer;
    const removeFlowAnimation = viewer.scene.preUpdate.addEventListener(() => {
      const seconds = 1 / 60;
      for (const particle of flowParticlesRef.current) {
        if (!flowEnabledRef.current) {
          particle.point.show = false;
          continue;
        }
        particle.point.show = true;
        const latitudeScale = Math.max(0.2, Math.cos(CesiumMath.toRadians(particle.lat)));
        particle.lon += (particle.velocityX * seconds * flowSpeedMultiplierRef.current) / (111_320 * latitudeScale);
        particle.lat += (particle.velocityY * seconds * flowSpeedMultiplierRef.current) / 111_320;
        if (
          particle.lon < rectFor(flowBoundsRef.current.longitude, flowBoundsRef.current.latitude).west ||
          particle.lon > rectFor(flowBoundsRef.current.longitude, flowBoundsRef.current.latitude).east ||
          particle.lat < rectFor(flowBoundsRef.current.longitude, flowBoundsRef.current.latitude).south ||
          particle.lat > rectFor(flowBoundsRef.current.longitude, flowBoundsRef.current.latitude).north
        ) {
          particle.lon = particle.originLon;
          particle.lat = particle.originLat;
        }
        particle.point.position = Cartesian3.fromDegrees(
          particle.lon,
          particle.lat,
          particle.elevation + particle.depth + 0.12
        );
      }
    });
    return () => {
      removeFlowAnimation();
      flowParticlesRef.current = [];
      flowPointsRef.current = null;
      rainSystemRef.current = null;
      buildingsRef.current = null;
      imageryLayerRef.current = null;
      viewer.destroy();
      viewerRef.current = null;
    };
  }, []);

  // --- Optional precipitation radar imagery -------------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !radarEnabled || !radarUrl) {
      if (viewer && radarLayerRef.current) {
        viewer.imageryLayers.remove(radarLayerRef.current, true);
        radarLayerRef.current = null;
      }
      return;
    }

    let radarLayer: ImageryLayer | null = null;
    let frameIndex = 0;
    const showRadarFrame = () => {
      if (radarLayer) viewer.imageryLayers.remove(radarLayer, true);
      const time = radarTimes[frameIndex] ?? '';
      radarLayer = new ImageryLayer(
        new UrlTemplateImageryProvider({
          url: radarUrl.replace('{time}', String(time)),
          credit: 'Weather radar provider',
          maximumLevel: 12,
        })
      );
      radarLayer.alpha = radarOpacity;
      viewer.imageryLayers.add(radarLayer);
      radarLayerRef.current = radarLayer;
      frameIndex = (frameIndex + 1) % Math.max(1, radarTimes.length);
    };
    showRadarFrame();
    const interval = radarTimes.length > 1 ? window.setInterval(showRadarFrame, 1_000) : null;

    return () => {
      if (interval !== null) window.clearInterval(interval);
      if (radarLayer) viewer.imageryLayers.remove(radarLayer, true);
      if (radarLayerRef.current === radarLayer) radarLayerRef.current = null;
    };
  }, [radarEnabled, radarOpacity, radarUrl, radarTimes]);

  // --- Rain particles ------------------------------------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (rainSystemRef.current) {
      viewer.scene.primitives.remove(rainSystemRef.current);
      rainSystemRef.current = null;
    }
    if (!rainEnabled || rainfallMmHr === null || rainfallMmHr <= 0) return;

    const rainRate = Math.min(140, Math.max(4, rainfallMmHr * 1.8));
    const windRadians = CesiumMath.toRadians(windDirection);
    const windEast = Math.sin(windRadians) * windSpeed;
    const windNorth = Math.cos(windRadians) * windSpeed;
    const rainSystem = new ParticleSystem({
      image: createRainTexture(),
      imageSize: new Cartesian2(5, 24),
      emissionRate: rainRate,
      minimumParticleLife: 0.7,
      maximumParticleLife: 1.4,
      minimumSpeed: 16,
      maximumSpeed: 24,
      startScale: 0.8,
      endScale: 1,
      startColor: Color.fromCssColorString('#c9f3ff').withAlpha(0.45),
      endColor: Color.fromCssColorString('#ffffff').withAlpha(0.05),
      emitter: new BoxEmitter(new Cartesian3(12_000, 12_000, 80)),
      modelMatrix: Matrix4.fromTranslation(
        Cartesian3.fromDegrees(longitude, latitude, 450)
      ),
      updateCallback: (particle) => {
        particle.velocity = new Cartesian3(windEast, windNorth, -22);
      },
    });
    viewer.scene.primitives.add(rainSystem);
    rainSystemRef.current = rainSystem;

    return () => {
      if (rainSystemRef.current === rainSystem) rainSystemRef.current = null;
      if (!rainSystem.isDestroyed()) viewer.scene.primitives.remove(rainSystem);
    };
  }, [rainEnabled, rainfallMmHr, longitude, latitude, windDirection, windSpeed]);

  // --- Optional Cesium Ion terrain + OSM buildings ------------------------
  useEffect(() => {
    if (!ionToken) return;
    const viewer = viewerRef.current;
    if (!viewer) return;
    let cancelled = false;
    (async () => {
      try {
        Ion.defaultAccessToken = ionToken;
        const imageryLayer = ImageryLayer.fromProviderAsync(IonImageryProvider.fromAssetId(2));
        viewer.imageryLayers.add(imageryLayer);
        imageryLayerRef.current = imageryLayer;
        const terrainProvider = await CesiumTerrainProvider.fromIonAssetId(1);
        if (cancelled) return;
        viewer.terrainProvider = terrainProvider;
        const buildings = await createOsmBuildingsAsync();
        if (cancelled) return;
        viewer.scene.primitives.add(buildings);
        buildingsRef.current = buildings;
      } catch (err) {
        console.warn('VARUNA: Cesium Ion unavailable, using fallback imagery/terrain.', err);
      }
    })();
    return () => {
      cancelled = true;
      if (imageryLayerRef.current) {
        viewer.imageryLayers.remove(imageryLayerRef.current, true);
        imageryLayerRef.current = null;
      }
      if (buildingsRef.current && !buildingsRef.current.isDestroyed()) {
        viewer.scene.primitives.remove(buildingsRef.current);
        buildingsRef.current = null;
      }
    };
  }, [ionToken]);

  // --- Click-to-inspect: report the cell under the cursor ------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas);
    handler.setInputAction((movement: ScreenSpaceEventHandler.PositionedEvent) => {
      if (!onPick || !terrain) {
        onPick?.(null);
        return;
      }
      const carto = viewer.camera.pickEllipsoid(
        movement.position,
        viewer.scene.globe.ellipsoid
      );
      if (!carto) {
        onPick(null);
        return;
      }
      const lonLat = Cartographic.fromCartesian(carto);
      if (!lonLat) {
        onPick(null);
        return;
      }
      const lon = CesiumMath.toDegrees(lonLat.longitude);
      const lat = CesiumMath.toDegrees(lonLat.latitude);
      const rect = rectFor(longitude, latitude);
      if (lon < rect.west || lon > rect.east || lat < rect.south || lat > rect.north) {
        onPick(null);
        return;
      }
      // Map position back to grid indices (row 0 = north, matching the DEM)
      const col = Math.min(
        terrain.size - 1,
        Math.max(0, Math.floor(((lon - rect.west) / (rect.east - rect.west)) * terrain.size))
      );
      const row = Math.min(
        terrain.size - 1,
        Math.max(0, Math.floor(((rect.north - lat) / (rect.north - rect.south)) * terrain.size))
      );
      const idx = row * terrain.size + col;
      onPick({
        lon,
        lat,
        elevation: terrain.elevation[idx],
        depth: frame ? frame.depths[idx] : 0,
      });
    }, ScreenSpaceEventType.LEFT_CLICK);
    return () => {
      handler.destroy();
    };
  }, [onPick, terrain, frame, longitude, latitude]);

  // --- Fly camera between the global Earth and study-area presets ----------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;
    flyToPreset(viewer, cameraPreset, longitude, latitude, cameraHeight);
  }, [cameraPreset, longitude, latitude, cameraHeight]);

  // --- Synthetic DEM tile ---------------------------------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !terrain) return;

    const rect = rectFor(longitude, latitude);
    const geometry = buildGridGeometry(terrain, rect, null, undefined, undefined, {
      skirt: true,
    });
    if (!geometry) return;

    const primitive = new Primitive({
      geometryInstances: new GeometryInstance({
        geometry,
        attributes: {
          color: ColorGeometryInstanceAttribute.fromColor(Color.fromCssColorString('#16324a')),
        },
      }),
      appearance: new PerInstanceColorAppearance({
        flat: false,
        translucent: false,
        closed: false,
      }),
      asynchronous: false,
    });
    viewer.scene.primitives.add(primitive);
    terrainPrimitiveRef.current = primitive;

    return () => {
      terrainPrimitiveRef.current = safeRemove(viewer, terrainPrimitiveRef.current);
    };
  }, [terrain, longitude, latitude]);

  // --- Flood surface, rebuilt per active frame -----------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !terrain || !frame) {
      if (viewer) floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);
      return;
    }

    floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);

    const rect = rectFor(longitude, latitude);

    // Constant 8 cm lift keeps shallow water from z-fighting the terrain skin;
    // real depth variation (up to metres) still dominates visually.
    const WATER_OFFSET_M = 0.08;

    const instances: GeometryInstance[] = [];
    for (let band = 0; band < BANDS.length; band++) {
      const geometry = buildGridGeometry(
        terrain,
        rect,
        frame.depths,
        bandIndexForDepth,
        band,
        { heightOffset: WATER_OFFSET_M }
      );
      if (!geometry) continue;
      const b = BANDS[band];
      instances.push(
        new GeometryInstance({
          geometry,
          attributes: {
            color: ColorGeometryInstanceAttribute.fromColor(
              Color.fromCssColorString(b.color).withAlpha(b.alpha)
            ),
          },
        })
      );
    }

    if (instances.length > 0) {
      const floodPrimitive = new Primitive({
        geometryInstances: instances,
        appearance: new PerInstanceColorAppearance({
          flat: true,
          translucent: true,
        }),
        asynchronous: false,
      });
      viewer.scene.primitives.add(floodPrimitive);
      floodPrimitiveRef.current = floodPrimitive;
    }

    return () => {
      floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);
    };
  }, [frame, terrain, longitude, latitude]);

  // --- Velocity-driven flow particles -------------------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (flowPointsRef.current) {
      viewer.scene.primitives.remove(flowPointsRef.current);
      flowPointsRef.current = null;
    }
    flowParticlesRef.current = [];

    if (
      !flowEnabled ||
      !terrain ||
      !frame?.velocityX ||
      !frame.velocityY ||
      frame.velocityX.length !== terrain.size * terrain.size ||
      frame.velocityY.length !== terrain.size * terrain.size
    ) {
      return;
    }

    const rect = rectFor(longitude, latitude);
    const collection = new PointPrimitiveCollection();
    const stride = Math.max(1, Math.floor(terrain.size / 12));
    for (let row = 0; row < terrain.size; row += stride) {
      for (let col = 0; col < terrain.size; col += stride) {
        const index = row * terrain.size + col;
        const depth = frame.depths[index];
        const velocityX = frame.velocityX[index];
        const velocityY = frame.velocityY[index];
        if (depth < 0.01 || Math.hypot(velocityX, velocityY) < 0.01) continue;

        const lon = rect.west + ((col + 0.5) / terrain.size) * (rect.east - rect.west);
        const lat = rect.north - ((row + 0.5) / terrain.size) * (rect.north - rect.south);
        const point = collection.add({
          position: Cartesian3.fromDegrees(lon, lat, terrain.elevation[index] + depth + 0.12),
          pixelSize: 4,
          color: Color.fromCssColorString('#b8ff24').withAlpha(0.8),
          disableDepthTestDistance: 0,
        });
        flowParticlesRef.current.push({
          point,
          lon,
          lat,
          velocityX,
          velocityY,
          elevation: terrain.elevation[index],
          depth,
          originLon: lon,
          originLat: lat,
        });
      }
    }

    if (flowParticlesRef.current.length > 0) {
      viewer.scene.primitives.add(collection);
      flowPointsRef.current = collection;
    }

    return () => {
      if (flowPointsRef.current === collection) flowPointsRef.current = null;
      viewer.scene.primitives.remove(collection);
      flowParticlesRef.current = [];
    };
  }, [flowEnabled, frame, terrain, longitude, latitude]);

  return <div ref={containerRef} className="h-full w-full" />;
}
