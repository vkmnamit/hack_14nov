import { useEffect, useRef, useState } from 'react';
import {
  Cartesian3,
  Cartesian2,
  Cartographic,
  Cesium3DTileset,
  Cesium3DTileStyle,
  BoxEmitter,
  Color,
  ColorGeometryInstanceAttribute,
  GeometryInstance,
  ImageryLayer,
  Matrix4,
  Ion,
  IonImageryProvider,
  Math as CesiumMath,
  Material,
  MaterialAppearance,
  PerInstanceColorAppearance,
  ParticleSystem,
  PointPrimitive,
  PointPrimitiveCollection,
  Polyline,
  PolylineCollection,
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
  onPick?: (info: {
    lon: number;
    lat: number;
    elevation: number;
    depth: number;
    velocity: number;
    flowDirection: number | null;
  } | null) => void;
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

const DEFAULT_DEPTH_EXAGGERATION = 4;
const WATER_OFFSET_M = 0.16;

function createAnimatedWaterMaterial(flowX = 0, flowY = 0, flowSpeed = 0): Material {
  const magnitude = Math.hypot(flowX, flowY);
  const direction = magnitude > 0.001 ? new Cartesian2(flowX / magnitude, flowY / magnitude) : new Cartesian2(0.7, 0.35);
  return new Material({
    fabric: {
      type: 'VARUNA_ANIMATED_WATER',
      uniforms: {
        color: Color.fromCssColorString('#087dbd').withAlpha(0.78),
        flowDirection: direction,
        flowSpeed: Math.min(2, Math.max(0.15, magnitude + flowSpeed * 0.15)),
      },
      source: `
        czm_material czm_getMaterial(czm_materialInput materialInput)
        {
          czm_material material = czm_getDefaultMaterial(materialInput);
          vec2 uv = materialInput.st;
          float phase = czm_frameNumber * 0.012 * flowSpeed;
          vec2 drift = flowDirection * phase;
          float waveA = sin((uv.x + drift.x) * 95.0 + (uv.y + drift.y) * 42.0);
          float waveB = sin((uv.x - drift.x * 0.65) * 48.0 - (uv.y - drift.y * 0.65) * 88.0);
          float ripple = 0.5 + 0.5 * (waveA * 0.58 + waveB * 0.42);
          float edge = smoothstep(0.12, 0.52, ripple);
          material.diffuse = color.rgb * (0.7 + ripple * 0.42);
          material.emission = color.rgb * (0.12 + edge * 0.12);
          material.alpha = color.a * (0.86 + ripple * 0.1);
          return material;
        }
      `,
    },
    translucent: true,
  });
}

function bandIndexForDepth(depth: number): number {
  for (let i = 0; i < BANDS.length; i++) if (depth < BANDS[i].max) return i;
  return BANDS.length - 1;
}

function averageVelocity(frame: FloodFrame): { x: number; y: number } {
  if (!frame.velocityX || !frame.velocityY) return { x: 0, y: 0 };
  let x = 0;
  let y = 0;
  let count = 0;
  for (let i = 0; i < frame.depths.length; i++) {
    if (frame.depths[i] < 0.01) continue;
    x += frame.velocityX[i] ?? 0;
    y += frame.velocityY[i] ?? 0;
    count++;
  }
  return count > 0 ? { x: x / count, y: y / count } : { x: 0, y: 0 };
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
  trail: Polyline;
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
  const waterAnimationPrimitiveRef = useRef<Primitive | null>(null);
  const buildingsRef = useRef<Cesium3DTileset | null>(null);
  const imageryLayerRef = useRef<ImageryLayer | null>(null);
  const radarLayerRef = useRef<ImageryLayer | null>(null);
  const rainSystemRef = useRef<ParticleSystem | null>(null);
  const flowPointsRef = useRef<PointPrimitiveCollection | null>(null);
  const flowTrailsRef = useRef<PolylineCollection | null>(null);
  const flowParticlesRef = useRef<FlowParticle[]>([]);
  const flowEnabledRef = useRef(flowEnabled);
  const flowSpeedMultiplierRef = useRef(flowSpeedMultiplier);
  const flowBoundsRef = useRef({ longitude, latitude });
  const depthExaggerationRef = useRef(DEFAULT_DEPTH_EXAGGERATION);
  const previousDepthsRef = useRef<number[] | null>(null);
  const floodTransitionRef = useRef<number | null>(null);
  const [depthExaggeration, setDepthExaggeration] = useState(DEFAULT_DEPTH_EXAGGERATION);

  useEffect(() => {
    flowEnabledRef.current = flowEnabled;
    flowSpeedMultiplierRef.current = flowSpeedMultiplier;
    flowBoundsRef.current = { longitude, latitude };
  }, [flowEnabled, flowSpeedMultiplier, longitude, latitude]);

  useEffect(() => {
    depthExaggerationRef.current = depthExaggeration;
  }, [depthExaggeration]);

  // --- One-time viewer bootstrapping --------------------------------------
  useEffect(() => {
    if (!containerRef.current) return;

    const baseLayer = new ImageryLayer(
      new UrlTemplateImageryProvider({
        url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
        credit: '© OpenStreetMap contributors',
      })
    );
    baseLayer.brightness = 1.18;
    baseLayer.contrast = 1.08;
    baseLayer.saturation = 1.12;
    baseLayer.gamma = 0.92;

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
      baseLayer,
    });

    viewer.scene.backgroundColor = Color.fromCssColorString('#030405');
    viewer.scene.globe.baseColor = Color.fromCssColorString('#080f16');
    viewer.scene.globe.enableLighting = true;
    viewer.scene.globe.showGroundAtmosphere = true;
    viewer.scene.fog.enabled = true;
    if (viewer.scene.skyAtmosphere) viewer.scene.skyAtmosphere.show = true;
    viewer.scene.screenSpaceCameraController.enableCollisionDetection = false;
    viewer.scene.globe.depthTestAgainstTerrain = true;
    viewer.scene.logarithmicDepthBuffer = true;
    viewer.scene.postProcessStages.fxaa.enabled = true;
    viewer.scene.msaaSamples = 4;
    viewer.camera.frustum.near = 1;
    viewer.camera.frustum.far = 50_000_000;
    viewer.shadows = false;
    viewer.scene.globe.baseColor = Color.fromCssColorString('#163447');

    viewerRef.current = viewer;
    const removeFlowAnimation = viewer.scene.preUpdate.addEventListener(() => {
      const seconds = 1 / 60;
      if (rainSystemRef.current) {
        const height = viewer.camera.positionCartographic.height;
        rainSystemRef.current.show = height < 750_000;
      }
      for (const particle of flowParticlesRef.current) {
        if (!flowEnabledRef.current) {
          particle.point.show = false;
          particle.trail.show = false;
          continue;
        }
        particle.point.show = true;
        particle.trail.show = true;
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
          particle.elevation + particle.depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08
        );
        particle.trail.positions = [
          Cartesian3.fromDegrees(
            particle.lon,
            particle.lat,
            particle.elevation + particle.depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08
          ),
          Cartesian3.fromDegrees(
            particle.lon - (particle.velocityX * 0.00006),
            particle.lat - (particle.velocityY * 0.00006),
            particle.elevation + particle.depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08
          ),
        ];
      }
    });
    return () => {
      removeFlowAnimation();
      flowParticlesRef.current = [];
      flowPointsRef.current = null;
      flowTrailsRef.current = null;
      rainSystemRef.current = null;
      waterAnimationPrimitiveRef.current = null;
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

    const rainRate = Math.min(240, Math.max(30, rainfallMmHr * 2.4));
    const windRadians = CesiumMath.toRadians(windDirection);
    const windEast = Math.sin(windRadians) * windSpeed;
    const windNorth = Math.cos(windRadians) * windSpeed;
    const rainSystem = new ParticleSystem({
      image: createRainTexture(),
      imageSize: new Cartesian2(8, 48),
      sizeInMeters: false,
      emissionRate: rainRate,
      minimumParticleLife: 1.4,
      maximumParticleLife: 2.4,
      minimumSpeed: 16,
      maximumSpeed: 24,
      startScale: 0.9,
      endScale: 1.15,
      startColor: Color.fromCssColorString('#c9f3ff').withAlpha(0.82),
      endColor: Color.fromCssColorString('#ffffff').withAlpha(0.18),
      emitter: new BoxEmitter(new Cartesian3(10_000, 10_000, 900)),
      modelMatrix: Matrix4.fromTranslation(
        Cartesian3.fromDegrees(longitude, latitude, 650)
      ),
      updateCallback: (particle) => {
        particle.velocity = new Cartesian3(windEast, windNorth, -22);
      },
    });
    rainSystem.show = viewer.camera.positionCartographic.height < 750_000;
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
        imageryLayer.brightness = 1.14;
        imageryLayer.contrast = 1.1;
        imageryLayer.saturation = 1.08;
        imageryLayer.gamma = 0.94;
        viewer.imageryLayers.add(imageryLayer);
        imageryLayerRef.current = imageryLayer;
        const terrainProvider = await CesiumTerrainProvider.fromIonAssetId(1);
        if (cancelled) return;
        viewer.terrainProvider = terrainProvider;
        const buildings = await createOsmBuildingsAsync();
        if (cancelled) return;
        buildings.style = new Cesium3DTileStyle({
          color: "color('#b9d1cf', 0.88)",
        });
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
      const velocityX = frame?.velocityX?.[idx] ?? 0;
      const velocityY = frame?.velocityY?.[idx] ?? 0;
      onPick({
        lon,
        lat,
        elevation: terrain.elevation[idx],
        depth: frame ? frame.depths[idx] : 0,
        velocity: Math.hypot(velocityX, velocityY),
        flowDirection: velocityX || velocityY ? (CesiumMath.toDegrees(Math.atan2(velocityX, velocityY)) + 360) % 360 : null,
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

  // --- Flood surface, interpolated between active frames -------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer || !terrain || !frame) {
      if (viewer) {
        floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);
        waterAnimationPrimitiveRef.current = safeRemove(viewer, waterAnimationPrimitiveRef.current);
      }
      previousDepthsRef.current = null;
      return;
    }

    const rect = rectFor(longitude, latitude);
    const targetDepths = frame.depths;
    const previousDepths = previousDepthsRef.current;
    const startDepths =
      previousDepths && previousDepths.length === targetDepths.length
        ? previousDepths
        : targetDepths;
    previousDepthsRef.current = [...targetDepths];

    const renderDepths = (depths: number[]) => {
      floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);
      waterAnimationPrimitiveRef.current = safeRemove(viewer, waterAnimationPrimitiveRef.current);

      const instances: GeometryInstance[] = [];
      for (let band = 0; band < BANDS.length; band++) {
        const geometry = buildGridGeometry(
          terrain,
          rect,
          depths,
          bandIndexForDepth,
          band,
          { heightOffset: WATER_OFFSET_M, depthScale: depthExaggerationRef.current }
        );
        if (!geometry) continue;
        const b = BANDS[band];
        instances.push(
          new GeometryInstance({
            geometry,
            attributes: {
              color: ColorGeometryInstanceAttribute.fromColor(
                Color.fromCssColorString(b.color).withAlpha(Math.max(0.78, b.alpha))
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

        const animatedGeometry = buildGridGeometry(
          terrain,
          rect,
          depths,
          () => 0,
          0,
          { heightOffset: WATER_OFFSET_M + 0.04, depthScale: depthExaggerationRef.current }
        );
        if (animatedGeometry) {
          const flow = averageVelocity(frame);
          const waterPrimitive = new Primitive({
            geometryInstances: new GeometryInstance({ geometry: animatedGeometry }),
            appearance: new MaterialAppearance({
              material: createAnimatedWaterMaterial(
                flow.x,
                flow.y,
                flowSpeedMultiplierRef.current
              ),
              materialSupport: MaterialAppearance.MaterialSupport.TEXTURED,
              faceForward: true,
              translucent: true,
            }),
            asynchronous: false,
          });
          viewer.scene.primitives.add(waterPrimitive);
          waterAnimationPrimitiveRef.current = waterPrimitive;
        }
      }
    };

    if (floodTransitionRef.current !== null) {
      cancelAnimationFrame(floodTransitionRef.current);
    }

    const startedAt = performance.now();
    let lastStep = -1;
    const animateTransition = (now: number) => {
      const progress = Math.min(1, (now - startedAt) / 420);
      const step = progress === 1 ? 6 : Math.floor(progress * 6);
      if (step !== lastStep) {
        lastStep = step;
        const eased = progress * progress * (3 - 2 * progress);
        const depths = targetDepths.map(
          (depth, index) => startDepths[index] + (depth - startDepths[index]) * eased
        );
        renderDepths(depths);
      }
      if (progress < 1) {
        floodTransitionRef.current = requestAnimationFrame(animateTransition);
      } else {
        floodTransitionRef.current = null;
      }
    };
    floodTransitionRef.current = requestAnimationFrame(animateTransition);

    return () => {
      if (floodTransitionRef.current !== null) {
        cancelAnimationFrame(floodTransitionRef.current);
        floodTransitionRef.current = null;
      }
      floodPrimitiveRef.current = safeRemove(viewer, floodPrimitiveRef.current);
      waterAnimationPrimitiveRef.current = safeRemove(viewer, waterAnimationPrimitiveRef.current);
    };
  }, [frame, terrain, longitude, latitude, depthExaggeration]);

  // --- Velocity-driven flow particles -------------------------------------
  useEffect(() => {
    const viewer = viewerRef.current;
    if (!viewer) return;

    if (flowPointsRef.current) {
      viewer.scene.primitives.remove(flowPointsRef.current);
      flowPointsRef.current = null;
    }
    if (flowTrailsRef.current) {
      viewer.scene.primitives.remove(flowTrailsRef.current);
      flowTrailsRef.current = null;
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
    const trails = new PolylineCollection();
    const stride = Math.max(1, Math.floor(terrain.size / 10));
    for (let row = 0; row < terrain.size; row += stride) {
      for (let col = 0; col < terrain.size; col += stride) {
        const index = row * terrain.size + col;
        const depth = frame.depths[index];
        const velocityX = frame.velocityX[index];
        const velocityY = frame.velocityY[index];
        const speed = Math.hypot(velocityX, velocityY);
        if (depth < 0.01 || speed < 0.01) continue;

        const lon = rect.west + ((col + 0.5) / terrain.size) * (rect.east - rect.west);
        const lat = rect.north - ((row + 0.5) / terrain.size) * (rect.north - rect.south);
        const point = collection.add({
          position: Cartesian3.fromDegrees(
            lon,
            lat,
            terrain.elevation[index] + depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08
          ),
          pixelSize: Math.min(18, 8 + speed * 7),
          color: Color.fromCssColorString(speed > 0.35 ? '#5da6ff' : '#00cfff').withAlpha(0.95),
          outlineColor: Color.WHITE.withAlpha(0.35),
          outlineWidth: 1,
          disableDepthTestDistance: Number.POSITIVE_INFINITY,
        });
        const trail = trails.add({
          positions: [
            Cartesian3.fromDegrees(lon, lat, terrain.elevation[index] + depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08),
            Cartesian3.fromDegrees(
              lon - velocityX * 0.00006,
              lat - velocityY * 0.00006,
              terrain.elevation[index] + depth * depthExaggerationRef.current + WATER_OFFSET_M + 0.08
            ),
          ],
          width: Math.min(4, 1.5 + speed * 2),
          material: Material.fromType('Color', {
            color: Color.fromCssColorString(speed > 0.35 ? '#5da6ff' : '#00cfff').withAlpha(0.7),
          }),
          show: true,
        });
        flowParticlesRef.current.push({
          point,
          trail,
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
      viewer.scene.primitives.add(trails);
      flowPointsRef.current = collection;
      flowTrailsRef.current = trails;
    }

    return () => {
      if (flowPointsRef.current === collection) flowPointsRef.current = null;
      if (flowTrailsRef.current === trails) flowTrailsRef.current = null;
      viewer.scene.primitives.remove(collection);
      viewer.scene.primitives.remove(trails);
      flowParticlesRef.current = [];
    };
  }, [flowEnabled, frame, terrain, longitude, latitude, depthExaggeration]);

  return (
    <div className="relative h-full w-full">
      <div ref={containerRef} className="h-full w-full" />
      <div className="pointer-events-auto absolute bottom-20 left-4 z-10 w-44 rounded-md border border-[var(--varuna-border)] bg-black/75 px-3 py-2 backdrop-blur-xl">
        <div className="mb-1 flex items-center justify-between gap-2">
          <label htmlFor="water-depth-exaggeration" className="text-[9px] uppercase tracking-[0.14em] text-[#9eb0a4]">
            Visual water depth
          </label>
          <span className="font-tabular text-[10px] text-white">{depthExaggeration.toFixed(1)}x</span>
        </div>
        <input
          id="water-depth-exaggeration"
          type="range"
          min="1"
          max="8"
          step="0.5"
          value={depthExaggeration}
          onChange={(event) => setDepthExaggeration(Number(event.target.value))}
          className="h-1 w-full accent-white"
          aria-label="Visual water depth exaggeration"
        />
        <p className="mt-1 text-[9px] text-[#7e8b96]">Display only; source depths unchanged.</p>
      </div>
    </div>
  );
}
