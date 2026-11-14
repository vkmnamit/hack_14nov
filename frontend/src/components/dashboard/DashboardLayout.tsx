import { useEffect, useRef, useState } from 'react';
import { Crosshair, Maximize2, MapPinned, Minimize2, PanelRightClose, RadioTower } from 'lucide-react';
import CesiumMap, { type CameraPreset } from '@/components/map/CesiumMap';
import SidebarControls from '@/components/dashboard/SidebarControls';
import AnalyticsPanel from '@/components/dashboard/AnalyticsPanel';
import OperationsSummary from '@/components/dashboard/OperationsSummary';
import ModelPerformancePanel from '@/components/dashboard/ModelPerformancePanel';
import SimulationTimeline from '@/components/dashboard/SimulationTimeline';
import FloodLegend from '@/components/dashboard/FloodLegend';
import TopBar, { getRuntimeIonToken } from '@/components/dashboard/TopBar';
import { SimulationProvider, useSimulation } from '@/state/SimulationContext';
import { cn } from '@/lib/utils';
import { RAIN_RADAR_TIMES, RAIN_RADAR_URL } from '@/config/appConfig';

interface CellReadout {
  lon: number;
  lat: number;
  elevation: number;
  depth: number;
  row: number;
  column: number;
  cellIndex: number;
  velocityX: number;
  velocityY: number;
  momentumX: number;
  momentumY: number;
  velocity: number;
  rainfallMmHr: number;
  cellWaterVolumeM3: number;
  buildingsAtRisk: number | null;
  roadWaysAtRisk: number | null;
  flowDirection: number | null;
}

function Dashboard() {
  const { state } = useSimulation();
  const [collapsed, setCollapsed] = useState(false);
  const [rightCollapsed, setRightCollapsed] = useState(false);
  const [activeView, setActiveView] = useState<'forecast' | 'performance'>('forecast');
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('global');
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [readout, setReadout] = useState<CellReadout | null>(null);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const frame = state.dataset?.frames[state.currentFrameIndex] ?? null;

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement === viewportRef.current);
    };
    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (state.location.id !== state.locations[0].id) setCameraPreset('study');
  }, [state.location.id, state.locations]);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) {
      void document.exitFullscreen();
      return;
    }
    if (viewportRef.current) void viewportRef.current.requestFullscreen();
  };

  const cameraPresets: { value: CameraPreset; label: string }[] = [
    { value: 'global', label: 'GLOBAL' },
    { value: 'india', label: 'INDIA' },
    { value: 'study', label: 'STUDY AREA' },
  ];

  return (
    <div ref={viewportRef} className={cn('city-shell flex h-screen w-screen flex-col overflow-hidden bg-[#05090b]', isFullscreen && 'immersive-shell')}>
      {!isFullscreen && (
        <>
          <TopBar
            simulationId={state.simulationMeta?.id ?? null}
            locationName={state.location.name}
            collapsed={collapsed}
            isDemoMode={state.isDemoMode}
            activeView={activeView}
            onViewChange={setActiveView}
            onToggleCollapse={() => {
              setCollapsed((c) => !c);
            }}
          />
          <OperationsSummary />
        </>
      )}

      <div className="relative min-h-0 flex-1 px-3 pb-3 lg:px-4">
        {/* Left sidebar */}
        <aside
          className={cn(
            'absolute inset-y-3 left-3 z-20 w-[232px] overflow-hidden rounded-md border border-[var(--varuna-border)] bg-black/80 transition-all duration-300 backdrop-blur-xl lg:left-4',
            collapsed || isFullscreen ? '-translate-x-full' : 'translate-x-0'
          )}
        >
          <SidebarControls />
        </aside>

        {/* Centre: 3D map + timeline */}
        <main className={cn(
          'relative flex h-full min-w-0 flex-1 flex-col transition-[margin] duration-300',
          !collapsed && 'lg:ml-[248px]',
          !rightCollapsed && 'md:mr-[294px]'
        )}>
          <div className="relative h-full min-h-0 flex-1 overflow-hidden rounded-md border border-[var(--varuna-border)] bg-black">
            <CesiumMap
              terrain={state.dataset?.terrain ?? null}
              frame={frame}
              frames={state.dataset?.frames ?? []}
              frameIndex={state.currentFrameIndex}
              longitude={state.location.longitude}
              latitude={state.location.latitude}
              cameraHeight={state.location.cameraHeight}
              ionToken={getRuntimeIonToken()}
              cameraPreset={cameraPreset}
              rainfallMmHr={frame?.rainfallMmHr ?? null}
              rainEnabled={state.rainEnabled}
              flowEnabled={state.flowEnabled}
              flowSpeedMultiplier={state.flowSpeedMultiplier}
              windDirection={state.windDirection}
              windSpeed={state.windSpeed}
              radarEnabled={state.radarEnabled}
              radarOpacity={state.radarOpacity}
              radarUrl={RAIN_RADAR_URL}
              radarTimes={RAIN_RADAR_TIMES}
              onPick={setReadout}
            />

            <div className="absolute left-4 top-4 z-10 flex items-center gap-1 rounded-md border border-[var(--varuna-border)] bg-black/75 p-1 backdrop-blur-xl">
              {cameraPresets.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  onClick={() => setCameraPreset(preset.value)}
                  className={cn(
                    'px-2 py-1.5 font-mono-num text-[9px] tracking-[0.08em] transition-colors',
                    cameraPreset === preset.value
                      ? 'bg-[#b8ff24] text-[#080b05] shadow-[0_0_18px_rgba(184,255,36,0.28)]'
                      : 'text-[#9a9a9a] hover:bg-white/[0.06] hover:text-white'
                  )}
                >
                  {preset.label}
                </button>
              ))}
            </div>

            <div className="absolute right-4 top-4 z-10 flex items-center gap-1 rounded-md border border-[var(--varuna-border)] bg-black/75 p-1 backdrop-blur-xl">
              <button
                type="button"
                onClick={() => setRightCollapsed((value) => !value)}
                className="flex h-7 w-7 items-center justify-center text-[#9a9a9a] transition-colors hover:bg-[#b8ff24]/10 hover:text-[#b8ff24]"
                aria-label={rightCollapsed ? 'Show analytics panel' : 'Hide analytics panel'}
              >
                <PanelRightClose size={14} />
              </button>
              <button
                type="button"
                onClick={toggleFullscreen}
                className="flex h-7 w-7 items-center justify-center text-[#9a9a9a] transition-colors hover:bg-[#b8ff24]/10 hover:text-[#b8ff24]"
                aria-label={isFullscreen ? 'Exit full screen' : 'Enter full screen'}
              >
                {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
            </div>

            {/* Legend overlay */}
            <div className="absolute bottom-4 right-4 z-10">
              <FloodLegend compact demo={state.isDemoMode} />
            </div>

            <div className="pointer-events-none absolute right-4 top-14 z-10 hidden w-64 rounded-md border border-[var(--varuna-border)] bg-black/75 p-3 backdrop-blur-xl md:block">
              <div className="mb-3 flex items-center justify-between border-b border-[var(--varuna-border)] pb-2">
                <div>
                  <p className="text-[9px] text-[#666]">Study area</p>
                  <p className="mt-1 text-sm font-medium text-white">{state.location.name}</p>
                </div>
                <MapPinned size={16} className="text-[#b8ff24]" />
              </div>
              <div className="grid grid-cols-2 gap-3 font-mono-num text-[10px]">
                <div>
                  <p className="text-[#666]">Region</p>
                  <p className="mt-1 text-white">{state.location.region}</p>
                </div>
                <div>
                  <p className="text-[#666]">Grid</p>
                  <p className="mt-1 text-white">{state.dataset?.terrain.size ?? '—'} × {state.dataset?.terrain.size ?? '—'}</p>
                </div>
                <div>
                  <p className="text-[#666]">Lat</p>
                  <p className="mt-1 text-[#cbd8cf]">{state.location.latitude.toFixed(4)}° N</p>
                </div>
                <div>
                  <p className="text-[#666]">Lon</p>
                  <p className="mt-1 text-[#cbd8cf]">{state.location.longitude.toFixed(4)}° E</p>
                </div>
              </div>
            </div>

            <div className="pointer-events-none absolute bottom-4 left-4 z-10 hidden items-center gap-2 rounded-md border border-[var(--varuna-border)] bg-black/75 px-3 py-2 backdrop-blur-xl sm:flex">
              <RadioTower size={13} className="text-white" />
              <div>
                <p className="text-[9px] text-white">Urban sensor mesh</p>
                <p className="text-[10px] text-[#9a9a9a]">
                  {state.isDemoMode ? 'Synthetic demo terrain' : 'Backend terrain • source reported by API'}
                </p>
              </div>
            </div>

            <div className="absolute right-4 top-32 z-10 flex flex-col gap-1 rounded-md border border-[var(--varuna-border)] bg-black/75 px-2.5 py-2 font-tabular text-[9px] backdrop-blur-xl">
              <span className={state.rainEnabled && (frame?.rainfallMmHr ?? 0) > 0 ? 'text-[#65c9e7]' : 'text-[#7e8b96]'}>
                {state.rainEnabled && (frame?.rainfallMmHr ?? 0) > 0
                  ? `RAIN ${frame?.rainfallMmHr?.toFixed(0)} MM/HR`
                  : 'RAIN DATA UNAVAILABLE'}
              </span>
              <span className={state.flowEnabled && frame?.velocityX && frame.velocityY ? 'text-white' : 'text-[#d7a12d]'}>
                {state.flowEnabled && frame?.velocityX && frame.velocityY
                  ? 'FLOW DATA AVAILABLE'
                  : 'FLOW DATA UNAVAILABLE'}
              </span>
              <span className={state.radarEnabled && RAIN_RADAR_URL ? 'text-[#00cfff]' : 'text-[#7e8b96]'}>
                {state.radarEnabled && RAIN_RADAR_URL ? 'RADAR OVERLAY ACTIVE' : 'RADAR UNAVAILABLE'}
              </span>
            </div>

            {/* Click-to-inspect table */}
            {readout && (
              <section className="absolute left-4 top-14 z-20 max-h-[65%] w-[min(360px,calc(100%-2rem))] overflow-y-auto rounded-md border border-[#31505a] bg-[#071014]/95 shadow-2xl backdrop-blur-xl">
                <header className="flex items-center justify-between border-b border-white/10 px-3 py-2">
                  <p className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-white">
                    <Crosshair size={13} className="text-[#b8ff24]" /> Grid cell details
                  </p>
                  <button type="button" onClick={() => setReadout(null)} className="px-1 text-lg leading-none text-slate-400 hover:text-white" aria-label="Close grid details">×</button>
                </header>
                <div className="border-b border-white/10 px-3 py-2 font-mono-num text-[10px] text-slate-400">
                  Cell {readout.cellIndex} · Row {readout.row + 1}, Column {readout.column + 1} · T+{frame?.t ?? 0} min
                </div>
                <div className="grid grid-cols-2 gap-x-4 px-3 py-2 font-mono-num text-[11px]">
                  <span className="text-slate-500">Lat</span>
                  <span className="text-right text-slate-200">{readout.lat.toFixed(5)}</span>
                  <span className="text-slate-500">Lon</span>
                  <span className="text-right text-slate-200">{readout.lon.toFixed(5)}</span>
                  <span className="text-slate-500">Elev</span>
                  <span className="text-right text-slate-200">{readout.elevation.toFixed(2)} m</span>
                  <span className="text-slate-500">Depth</span>
                  <span
                    className={cn(
                      'text-right font-semibold',
                      readout.depth >= 0.5
                        ? 'text-[#ed7044]'
                        : readout.depth >= 0.3
                          ? 'text-[#f4ca35]'
                          : readout.depth >= 0.01
                            ? 'text-[#65c9e7]'
                            : 'text-slate-500'
                    )}
                  >
                    {(readout.depth * 100).toFixed(1)} cm
                  </span>
                  <span className="text-slate-500">Water surface</span>
                  <span className="text-right text-slate-200">{(readout.elevation + readout.depth).toFixed(2)} m</span>
                  <span className="text-slate-500">Cell water volume</span>
                  <span className="text-right text-slate-200">{readout.cellWaterVolumeM3.toFixed(2)} m³</span>
                  <span className="text-slate-500">Rainfall</span>
                  <span className="text-right text-slate-200">{readout.rainfallMmHr.toFixed(1)} mm/hr</span>
                  <span className="text-slate-500">Velocity east (u)</span>
                  <span className="text-right text-[#5da6ff]">{readout.velocityX.toFixed(4)} m/s</span>
                  <span className="text-slate-500">Velocity north (v)</span>
                  <span className="text-right text-[#5da6ff]">{readout.velocityY.toFixed(4)} m/s</span>
                  <span className="text-slate-500">Flow speed</span>
                  <span className="text-right text-[#5da6ff]">{readout.velocity.toFixed(4)} m/s</span>
                  <span className="text-slate-500">Direction</span>
                  <span className="text-right text-[#5da6ff]">{readout.flowDirection === null ? '—' : `${readout.flowDirection.toFixed(1)}°`}</span>
                  <span className="text-slate-500">X momentum (hu)</span>
                  <span className="text-right text-slate-200">{readout.momentumX.toFixed(5)} m²/s</span>
                  <span className="text-slate-500">Y momentum (hv)</span>
                  <span className="text-right text-slate-200">{readout.momentumY.toFixed(5)} m²/s</span>
                  <span className="text-slate-500">Buildings at risk in cell</span>
                  <span className="text-right text-slate-200">
                    {readout.buildingsAtRisk === null
                      ? state.isDemoMode
                        ? 'Demo only'
                        : 'OSM unavailable'
                      : readout.buildingsAtRisk}
                  </span>
                  <span className="text-slate-500">Road ways at risk in cell</span>
                  <span className="text-right text-slate-200">
                    {readout.roadWaysAtRisk === null
                      ? state.isDemoMode
                        ? 'Demo only'
                        : 'OSM unavailable'
                      : readout.roadWaysAtRisk}
                  </span>
                  <span className="text-slate-500">Risk</span>
                  <span className={cn('text-right font-semibold', readout.depth >= 0.5 || readout.velocity >= 1.5 ? 'text-red-400' : readout.depth >= 0.3 || readout.velocity >= 0.8 ? 'text-amber-300' : readout.depth >= 0.01 ? 'text-cyan-300' : 'text-slate-400')}>
                    {readout.depth >= 0.5 || readout.velocity >= 1.5 ? 'High' : readout.depth >= 0.3 || readout.velocity >= 0.8 ? 'Moderate' : readout.depth >= 0.01 ? 'Low' : 'Dry'}
                  </span>
                </div>
              </section>
            )}

            {/* Loading overlay while a simulation runs */}
            {state.status === 'running' && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-black/70 backdrop-blur-sm">
                <div className="w-72 rounded-md border border-[var(--varuna-border-strong)] bg-[#050505] p-5 text-center">
                  <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-[#1b2840] border-t-[#00D4FF]" />
                  <p className="mt-3 text-sm font-medium text-slate-200">Computing flood frames</p>
                  <p className="mt-1 text-xs text-slate-500">
                    Solving depth field over the {state.dataset?.terrain.size ?? 'configured'} grid
                  </p>
                  <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[#1b2840]">
                    <div
                      className="h-full rounded-full bg-[var(--varuna-blue-2)] transition-all duration-300"
                      style={{ width: `${state.progress}%` }}
                    />
                  </div>
                  <p className="mt-2 font-mono-num text-xs text-[#00D4FF]">{state.progress}%</p>
                </div>
              </div>
            )}
          </div>

          <SimulationTimeline className="absolute bottom-0 left-0 right-0 z-10" />
        </main>

        {/* Right sidebar */}
        <aside className={cn(
          'absolute inset-y-3 right-3 z-20 hidden w-[278px] overflow-hidden rounded-md border border-[var(--varuna-border)] bg-black/80 transition-all duration-300 backdrop-blur-xl md:right-4 md:block',
          (rightCollapsed || isFullscreen) && 'translate-x-full'
        )}>
          {activeView === 'forecast' ? <AnalyticsPanel /> : <ModelPerformancePanel />}
        </aside>
      </div>
    </div>
  );
}

export default function DashboardLayout() {
  return (
    <SimulationProvider>
      <Dashboard />
    </SimulationProvider>
  );
}
