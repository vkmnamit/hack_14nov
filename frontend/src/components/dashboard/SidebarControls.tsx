import { CloudRain, Droplets, Gauge, MapPin, Pause, Play, Radar, RotateCcw, Timer, Waves, Wind, Zap } from 'lucide-react';
import { Area, AreaChart } from 'recharts';
import { hyetograph } from '@/lib/demoData';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { useSimulation } from '@/state/SimulationContext';
import type { RainfallPattern, SimulationStatus } from '@/lib/demoData';
import { cn } from '@/lib/utils';
import { RAIN_RADAR_URL } from '@/config/appConfig';

const DURATION_OPTIONS = [15, 30, 60, 90];

const PATTERN_OPTIONS: { value: RainfallPattern; label: string }[] = [
  { value: 'constant', label: 'Constant' },
  { value: 'increasing', label: 'Increasing' },
  { value: 'peak', label: 'Peak-Intensity' },
];

const STATUS_STYLES: Record<SimulationStatus, string> = {
  ready: 'border-[#334238] bg-[#101613] text-[#aeb9b0]',
  running: 'border-[#65c9e7]/50 bg-[#65c9e7]/10 text-[#65c9e7]',
  paused: 'border-[#f4ca35]/50 bg-[#f4ca35]/10 text-[#f4ca35]',
  completed: 'border-[#9ad52d]/50 bg-[#9ad52d]/10 text-[#b7eb62]',
  error: 'border-[#ff3939]/50 bg-[#ff3939]/10 text-[#ff7777]',
};

function SectionTitle({ icon, children }: { icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2 border-b border-[#1b2821] pb-2">
      <span className="text-[#9ad52d]">{icon}</span>
      <h3 className="text-[11px] font-semibold uppercase tracking-widest text-[#e2e3e6]">
        {children}
      </h3>
    </div>
  );
}

export default function SidebarControls() {
  const {
    state,
    setLocation,
    setIntensity,
    setDuration,
    setPattern,
    runSimulation,
    pause,
    resume,
    reset,
    setRainEnabled,
    setFlowEnabled,
    setFlowSpeedMultiplier,
    setWindDirection,
    setWindSpeed,
    setRadarEnabled,
    setRadarOpacity,
  } = useSimulation();

  const isRunning = state.status === 'running';
  const hasResult = state.dataset !== null && state.dataset.frames.length > 0;

  return (
    <div className="panel-scroll flex h-full flex-col gap-4 overflow-y-auto bg-[#080b0c] p-3">
      {/* Location */}
      <section>
        <SectionTitle icon={<MapPin size={14} />}>Location</SectionTitle>
        <Select value={state.location.id} onValueChange={(id) => {
          const loc = state.locations.find((l) => l.id === id);
          if (loc) setLocation(loc);
        }}>
          <SelectTrigger className="w-full border-[#30453f] bg-[#111c1d] text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="border-[#30453f] bg-[#111c1d]">
            {state.locations.map((loc) => (
              <SelectItem key={loc.id} value={loc.id}>
                {loc.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="mt-1.5 text-[11px] leading-snug text-slate-500">{state.location.region}</p>
      </section>

      {/* Rainfall configuration */}
      <section>
        <SectionTitle icon={<Droplets size={14} />}>Rainfall</SectionTitle>

        <div className="mb-1 flex items-baseline justify-between">
          <Label className="text-xs text-slate-400">Intensity</Label>
          <span className="font-mono-num text-sm font-semibold text-[#b7eb62]">
            {state.rainfallIntensity} <span className="text-xs text-slate-400">mm/hr</span>
          </span>
        </div>
        <Slider
          value={[state.rainfallIntensity]}
          min={0}
          max={150}
          step={5}
          onValueChange={([v]) => setIntensity(v)}
          className="my-2 [&_[data-radix-slider-track]]:bg-[#30453f] [&_[data-radix-slider-range]]:bg-[#d7ff3f]"
          aria-label="Rainfall intensity in millimetres per hour"
        />

        <div className="mb-1 mt-4 flex items-baseline justify-between">
          <Label className="text-xs text-slate-400">Duration</Label>
          <span className="font-mono-num text-sm font-semibold text-[#b7eb62]">
            {state.durationMin} <span className="text-xs text-slate-400">min</span>
          </span>
        </div>
        <div className="grid grid-cols-4 gap-1.5">
          {DURATION_OPTIONS.map((d) => (
            <button
              key={d}
              onClick={() => setDuration(d)}
              className={cn(
                'rounded border px-1 py-1.5 font-mono-num text-xs transition-colors',
                state.durationMin === d
                  ? 'border-[#9ad52d]/70 bg-[#9ad52d]/12 text-[#b7eb62]'
                  : 'border-[#30453f] bg-[#111c1d] text-slate-400 hover:border-[#82958a] hover:text-slate-200'
              )}
            >
              {d}
            </button>
          ))}
        </div>

        <div className="mb-1 mt-4">
          <Label className="text-xs text-slate-400">Pattern</Label>
        </div>
        <Select value={state.pattern} onValueChange={(v) => setPattern(v as RainfallPattern)}>
          <SelectTrigger className="w-full border-[#30453f] bg-[#111c1d] text-sm">
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="border-[#30453f] bg-[#111c1d]">
            {PATTERN_OPTIONS.map((p) => (
              <SelectItem key={p.value} value={p.value}>
                {p.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        {/* Hyetograph preview: storm shape updates live with intensity, duration, pattern */}
        <div className="mt-3 rounded border border-[#263839] bg-[#111c1d] p-2">
          <p className="mb-1 text-[9px] font-medium uppercase tracking-wider text-slate-500">
            Storm profile
          </p>
          <div className="h-10">
            <AreaChart
              width={220}
              height={40}
              data={Array.from({ length: 25 }, (_, i) => ({
                t: (i / 24) * (state.durationMin + 20),
                r: hyetograph(state.pattern, state.rainfallIntensity, state.durationMin, (i / 24) * (state.durationMin + 20)),
              }))}
              margin={{ top: 2, right: 0, bottom: 0, left: 0 }}
            >
                <defs>
                  <linearGradient id="rainFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#b7eb62" stopOpacity={0.7} />
                    <stop offset="100%" stopColor="#b7eb62" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <Area
                  type="monotone"
                  dataKey="r"
                  stroke="#b7eb62"
                  strokeWidth={1.5}
                  fill="url(#rainFill)"
                  isAnimationActive={false}
                />
            </AreaChart>
          </div>
        </div>
      </section>

      {/* Visualization controls */}
      <section>
        <SectionTitle icon={<CloudRain size={14} />}>Atmosphere & flow</SectionTitle>
        <div className="space-y-2">
          {[
            { label: 'Rain particles', value: state.rainEnabled, onChange: setRainEnabled },
            { label: 'Flood flow particles', value: state.flowEnabled, onChange: setFlowEnabled },
            { label: 'Radar overlay', value: state.radarEnabled, onChange: setRadarEnabled, disabled: !RAIN_RADAR_URL },
          ].map((control) => (
            <button
              key={control.label}
              type="button"
              onClick={() => control.onChange(!control.value)}
              disabled={control.disabled}
              className={cn(
                'flex w-full items-center justify-between rounded border px-2.5 py-2 text-left text-[11px] transition-colors',
                control.value
                  ? 'border-[#9ad52d]/50 bg-[#9ad52d]/10 text-[#b7eb62]'
                  : 'border-[#30453f] bg-[#111c1d] text-slate-400',
                control.disabled && 'cursor-not-allowed opacity-45'
              )}
            >
              <span>{control.label}</span>
              <span className="font-mono-num text-[9px] uppercase">{control.disabled ? 'Not configured' : control.value ? 'On' : 'Off'}</span>
            </button>
          ))}
        </div>

        <div className="mt-3 rounded border border-[#263839] bg-[#111c1d] p-2.5">
          <div className="mb-1 flex items-center justify-between">
            <Label className="flex items-center gap-1.5 text-[10px] text-slate-400"><Wind size={12} /> Wind direction</Label>
            <span className="font-mono-num text-[10px] text-[#b7eb62]">{state.windDirection}°</span>
          </div>
          <Slider
            value={[state.windDirection]}
            min={0}
            max={360}
            step={5}
            onValueChange={([value]) => setWindDirection(value)}
            aria-label="Wind direction in degrees"
          />
          <div className="mb-1 mt-3 flex items-center justify-between">
            <Label className="text-[10px] text-slate-400">Wind speed</Label>
            <span className="font-mono-num text-[10px] text-[#b7eb62]">{state.windSpeed} m/s</span>
          </div>
          <Slider
            value={[state.windSpeed]}
            min={0}
            max={20}
            step={1}
            onValueChange={([value]) => setWindSpeed(value)}
            aria-label="Wind speed in metres per second"
          />
          <div className="mb-1 mt-3 flex items-center justify-between">
            <Label className="text-[10px] text-slate-400">Flow speed</Label>
            <span className="font-mono-num text-[10px] text-[#b7eb62]">{state.flowSpeedMultiplier.toFixed(1)}x</span>
          </div>
          <Slider
            value={[state.flowSpeedMultiplier]}
            min={0.25}
            max={4}
            step={0.25}
            onValueChange={([value]) => setFlowSpeedMultiplier(value)}
            aria-label="Flood flow speed multiplier"
          />
          {state.radarEnabled && RAIN_RADAR_URL && (
            <>
              <div className="mb-1 mt-3 flex items-center justify-between">
                <Label className="flex items-center gap-1.5 text-[10px] text-slate-400"><Radar size={12} /> Radar opacity</Label>
                <span className="font-mono-num text-[10px] text-[#00cfff]">{Math.round(state.radarOpacity * 100)}%</span>
              </div>
              <Slider
                value={[state.radarOpacity]}
                min={0.1}
                max={1}
                step={0.05}
                onValueChange={([value]) => setRadarOpacity(value)}
                aria-label="Radar opacity"
              />
            </>
          )}
        </div>
        <p className="mt-2 text-[9px] leading-relaxed text-slate-600">
          Rain is visualized from the active frame. Flow particles require backend or demo velocity fields.
        </p>
      </section>

      {/* Simulation controls */}
      <section>
        <SectionTitle icon={<Gauge size={14} />}>Simulation</SectionTitle>

        <div className="grid grid-cols-3 gap-1.5">
          <Button
            onClick={runSimulation}
            disabled={isRunning}
            className="h-9 bg-[#d7ff3f] px-2 text-[#11180d] hover:bg-[#efffb2] hover:text-[#11180d]"
            title="Run the flood simulation with the current settings"
          >
            <Play size={14} className="mr-1" /> Run
          </Button>
          <Button
            onClick={state.playing ? pause : resume}
            disabled={!hasResult || state.status === 'ready'}
            variant="outline"
            className="h-9 border-[#30453f] bg-[#111c1d] px-2 text-slate-200 hover:border-[#82958a] hover:bg-[#172424]"
            title="Pause or resume the simulation playback"
          >
            {state.playing ? <Pause size={14} className="mr-1" /> : <Play size={14} className="mr-1" />}
            {state.playing ? 'Pause' : 'Resume'}
          </Button>
          <Button
            onClick={reset}
            variant="outline"
            className="h-9 border-[#30453f] bg-[#111c1d] px-2 text-slate-200 hover:border-[#82958a] hover:bg-[#172424]"
            title="Reset the simulation and clear results"
          >
            <RotateCcw size={14} className="mr-1" /> Reset
          </Button>
        </div>

        {/* Progress */}
        <div className="mt-4">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[10px] font-medium uppercase tracking-wider text-slate-400">
              Progress
            </span>
            <span
              className={cn(
                'rounded border px-1.5 py-0.5 font-mono-num text-[10px] font-semibold uppercase tracking-wider',
                STATUS_STYLES[state.status],
                state.status === 'running' && 'animate-status-pulse'
              )}
            >
              {state.status}
            </span>
          </div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#202b24]">
            <div
              className="h-full rounded-full bg-gradient-to-r from-[#6aa91b] to-[#b7eb62] transition-all duration-300"
              style={{ width: `${state.progress}%` }}
            />
          </div>
          {state.errorMessage && (
            <p className="mt-2 text-[10px] leading-relaxed text-[#ff7777]" role="alert">
              {state.errorMessage}
            </p>
          )}
        </div>

        {hasResult && state.simulationMeta && (
          <div className="mt-4 rounded border border-[#263839] bg-[#111c1d] p-2.5">
            <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-slate-500">
              <Timer size={11} /> Simulation
            </div>
            <p className="mt-1 truncate font-mono-num text-xs text-[#e2e3e6]">
              {state.simulationMeta.id}
            </p>
            <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 font-mono-num text-[11px]">
              <span className="text-slate-500">Rain</span>
              <span className="text-right text-[#e2e3e6]">{state.rainfallIntensity} mm/hr</span>
              <span className="text-slate-500">Duration</span>
              <span className="text-right text-[#e2e3e6]">{state.durationMin} min</span>
              <span className="text-slate-500">Pattern</span>
              <span className="text-right capitalize text-[#e2e3e6]">{state.pattern}</span>
              <span className="flex items-center gap-1 text-slate-500">
                <Waves size={10} /> Frames
              </span>
              <span className="text-right text-[#e2e3e6]">{state.dataset?.frames.length ?? 0}</span>
            </div>
          </div>
        )}
      </section>

      <div className="mt-auto rounded border border-[#263839] bg-[#111c1d] p-3">
        <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-[#f4ca35]">
          <Zap size={11} /> {state.isDemoMode ? 'Demo mode' : 'Connected mode'}
        </div>
        <p className="mt-1 text-[10px] leading-relaxed text-slate-500">
          {state.isDemoMode
            ? 'Running synthetic terrain and hydrology. Results are illustrative only and are not real flood predictions.'
            : 'Simulation requests use the configured backend. Results remain unavailable until the backend returns valid terrain and frame data.'}
        </p>
      </div>
    </div>
  );
}
