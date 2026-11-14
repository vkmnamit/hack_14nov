import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Crosshair, Maximize2, MapPinned, Minimize2, PanelRightClose, RadioTower } from 'lucide-react';
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
    { value: 'bengaluru', label: 'BENGALURU' },
    { value: 'study', label: 'STUDY AREA' },
  ];

  return (
    <div ref={viewportRef} className={cn('city-shell flex h-screen w-screen flex-col overflow-hidden bg-[#05090b]', isFullscreen && 'immersive-shell')}>
      {!isFullscreen && (
        <>
          <TopBar
            simulationId={state.simulationMeta?.id ?? null}
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
            'absolute inset-y-3 left-3 z-20 w-[232px] overflow-hidden rounded-2xl border border-[#26332c] bg-[#080b0f]/95 shadow-[0_18px_45px_rgba(0,0,0,0.42)] transition-all duration-300 backdrop-blur-xl lg:left-4',
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
          <div className="relative h-full min-h-0 flex-1 overflow-hidden rounded-2xl border border-[#26332c] bg-[#05090b] shadow-[0_18px_45px_rgba(0,0,0,0.4)]">
            <CesiumMap
              terrain={state.dataset?.terrain ?? null}
              frame={frame}
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

            <div className="absolute left-4 top-4 z-10 flex items-center gap-1 rounded-sm border border-[#202b34] bg-[#080b0f]/90 p-1 shadow-xl backdrop-blur-xl">
              {cameraPresets.map((preset) => (
                <button
                  key={preset.value}
                  type="button"
                  onClick={() => setCameraPreset(preset.value)}
                  className={cn(
                    'px-2 py-1.5 font-mono-num text-[9px] tracking-[0.08em] transition-colors',
                    cameraPreset === preset.value
                      ? 'bg-[#b8ff24] text-[#030405]'
                      : 'text-[#7e8b96] hover:bg-[#111820] hover:text-[#e8edf2]'
                  )}
                >
                  {preset.label}
                </button>
              ))}
            </div>

            <div className="absolute right-4 top-4 z-10 flex items-center gap-1 rounded-sm border border-[#202b34] bg-[#080b0f]/90 p-1 shadow-xl backdrop-blur-xl">
              <button
                type="button"
                onClick={() => setRightCollapsed((value) => !value)}
                className="flex h-7 w-7 items-center justify-center text-[#7e8b96] transition-colors hover:bg-[#111820] hover:text-[#b8ff24]"
                aria-label={rightCollapsed ? 'Show analytics panel' : 'Hide analytics panel'}
              >
                <PanelRightClose size={14} />
              </button>
              <button
                type="button"
                onClick={toggleFullscreen}
                className="flex h-7 w-7 items-center justify-center text-[#7e8b96] transition-colors hover:bg-[#111820] hover:text-[#b8ff24]"
                aria-label={isFullscreen ? 'Exit full screen' : 'Enter full screen'}
              >
                {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
            </div>

            {/* Legend overlay */}
            <div className="absolute bottom-4 right-4 z-10">
              <FloodLegend compact demo={state.isDemoMode} />
            </div>

            <div className="pointer-events-none absolute right-4 top-4 z-10 hidden w-64 rounded-sm border border-[#536653] bg-[#0b1415]/85 p-3 shadow-[0_12px_35px_rgba(0,0,0,0.42)] backdrop-blur-xl md:block">
              <div className="mb-3 flex items-center justify-between border-b border-[#263839] pb-2">
                <div>
                  <p className="text-[9px] uppercase tracking-[0.2em] text-[#82958a]">Study area</p>
                  <p className="mt-1 text-sm font-medium text-[#f0f1f3]">{state.location.name}</p>
                </div>
                <MapPinned size={16} className="text-[#d7ff3f]" />
              </div>
              <div className="grid grid-cols-2 gap-3 font-mono-num text-[10px]">
                <div>
                  <p className="text-[#82958a]">ZONE</p>
                  <p className="mt-1 text-[#d7ff3f]">BLR-EAST-01</p>
                </div>
                <div>
                  <p className="text-[#82958a]">GRID</p>
                  <p className="mt-1 text-[#d7ff3f]">{state.dataset?.terrain.size ?? '—'} × {state.dataset?.terrain.size ?? '—'}</p>
                </div>
                <div>
                  <p className="text-[#82958a]">LAT</p>
                  <p className="mt-1 text-[#cbd8cf]">{state.location.latitude.toFixed(4)}° N</p>
                </div>
                <div>
                  <p className="text-[#82958a]">LON</p>
                  <p className="mt-1 text-[#cbd8cf]">{state.location.longitude.toFixed(4)}° E</p>
                </div>
              </div>
            </div>

            <div className="pointer-events-none absolute bottom-4 left-4 z-10 hidden items-center gap-2 rounded-sm border border-[#536653] bg-[#0b1415]/85 px-3 py-2 shadow-lg backdrop-blur-xl sm:flex">
              <RadioTower size={13} className="text-[#d7ff3f]" />
              <div>
                <p className="font-mono-num text-[9px] uppercase tracking-[0.16em] text-[#d7ff3f]">Urban sensor mesh</p>
                <p className="text-[10px] text-[#9eb0a4]">
                  {state.isDemoMode ? 'Synthetic terrain • Bengaluru, Karnataka' : 'Backend terrain • source reported by API'}
                </p>
              </div>
            </div>

            {/* Data source status */}
            <div className="absolute left-4 top-14 z-10 flex items-center gap-1.5 rounded-sm border border-[#202b34] bg-[#080b0f]/90 px-2.5 py-1.5 shadow-lg backdrop-blur-xl">
              <AlertTriangle size={12} className="shrink-0 text-[#f4ca35]" />
              <span className="text-[10px] font-medium tracking-wide text-[#f4ca35]">
                {state.isDemoMode
                  ? 'DEMO DATA — synthetic terrain, not real forecasts'
                  : 'CONNECTED MODE — backend data required'}
              </span>
            </div>

            <div className="absolute right-4 top-20 z-10 flex flex-col gap-1 rounded-sm border border-[#202b34] bg-[#080b0f]/90 px-2.5 py-2 font-mono-num text-[9px] uppercase tracking-[0.12em] shadow-lg backdrop-blur-xl">
              <span className={state.rainEnabled && (frame?.rainfallMmHr ?? 0) > 0 ? 'text-[#65c9e7]' : 'text-[#7e8b96]'}>
                {state.rainEnabled && (frame?.rainfallMmHr ?? 0) > 0
                  ? `RAIN ${frame?.rainfallMmHr?.toFixed(0)} MM/HR`
                  : 'RAIN DATA UNAVAILABLE'}
              </span>
              <span className={state.flowEnabled && frame?.velocityX && frame.velocityY ? 'text-[#b8ff24]' : 'text-[#ffb547]'}>
                {state.flowEnabled && frame?.velocityX && frame.velocityY
                  ? 'FLOW DATA AVAILABLE'
                  : 'FLOW DATA UNAVAILABLE'}
              </span>
              <span className={state.radarEnabled && RAIN_RADAR_URL ? 'text-[#00cfff]' : 'text-[#7e8b96]'}>
                {state.radarEnabled && RAIN_RADAR_URL ? 'RADAR OVERLAY ACTIVE' : 'RADAR UNAVAILABLE'}
              </span>
            </div>

            {/* Click-to-inspect readout */}
            <div
              className={cn(
                'absolute left-4 top-12 z-10 rounded-sm border border-[#646772] bg-[#363843]/95 px-3 py-2 shadow-lg backdrop-blur transition-opacity duration-200',
                readout ? 'opacity-100' : 'pointer-events-none opacity-0'
              )}
            >
              <p className="mb-1 flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-widest text-slate-400">
                <Crosshair size={10} className="text-[#00D4FF]" /> Cell inspector
              </p>
              {readout && (
                <div className="grid grid-cols-2 gap-x-4 gap-y-0.5 font-mono-num text-[11px]">
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
                </div>
              )}
            </div>

            {/* Loading overlay while a simulation runs */}
            {state.status === 'running' && (
              <div className="absolute inset-0 z-20 flex items-center justify-center bg-[#252730]/65 backdrop-blur-sm">
                <div className="w-72 rounded-sm border border-[#5c5f6b] bg-[#3a3c46] p-5 text-center shadow-2xl">
                  <div className="mx-auto h-10 w-10 animate-spin rounded-full border-2 border-[#1b2840] border-t-[#00D4FF]" />
                  <p className="mt-3 text-sm font-medium text-slate-200">Computing flood frames</p>
                  <p className="mt-1 text-xs text-slate-500">
                    Solving depth field over the {state.dataset?.terrain.size ?? 'configured'} grid
                  </p>
                  <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[#1b2840]">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-[#247BFF] to-[#00D4FF] transition-all duration-300"
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
          'absolute inset-y-3 right-3 z-20 hidden w-[278px] overflow-hidden rounded-2xl border border-[#26332c] bg-[#080b0f]/95 shadow-[0_18px_45px_rgba(0,0,0,0.42)] transition-all duration-300 backdrop-blur-xl md:right-4 md:block',
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
