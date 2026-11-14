import { useEffect } from 'react';
import { Pause, Play, SkipBack, SkipForward } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { useSimulation } from '@/state/SimulationContext';
import { cn } from '@/lib/utils';

const SPEEDS: (1 | 2 | 4)[] = [1, 2, 4];

interface SimulationTimelineProps {
  className?: string;
}

export default function SimulationTimeline({ className }: SimulationTimelineProps) {
  const { state, setFrame, setPlaying, setSpeed } = useSimulation();
  const frames = state.dataset?.frames ?? [];
  const maxIndex = Math.max(0, frames.length - 1);
  const frame = frames[state.currentFrameIndex] ?? null;
  const totalMin = frames.length > 0 ? frames[frames.length - 1].t : state.durationMin;
  const timelineMarks = Array.from(
    new Set([0, 15, 30, 45, 60, totalMin].filter((minute) => minute <= totalMin))
  ).sort((a, b) => a - b);

  useEffect(() => {
    if (!state.playing || maxIndex === 0) return;
    if (state.currentFrameIndex >= maxIndex) {
      setPlaying(false);
      return;
    }
    const id = window.setTimeout(() => setFrame(state.currentFrameIndex + 1), 600 / state.playbackSpeed);
    return () => window.clearTimeout(id);
  }, [state.playing, state.playbackSpeed, state.currentFrameIndex, maxIndex, setFrame, setPlaying]);

  const minutes = frame ? frame.t : 0;
  const timestamp = frame
    ? new Date(Date.now() + frame.t * 60000).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })
    : '--:--';

  return (
    <div className={cn('flex h-14 items-center gap-3 border-t border-[var(--varuna-border)] bg-black/85 px-4 backdrop-blur-xl', className)}>
      <button onClick={() => state.playing ? setPlaying(false) : setPlaying(true)} disabled={maxIndex === 0} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[#b8ff24] text-black shadow-[0_0_16px_rgba(184,255,36,0.3)] transition hover:bg-[#d7ff70] disabled:opacity-40" aria-label={state.playing ? 'Pause playback' : 'Play playback'}>
        {state.playing ? <Pause size={16} /> : <Play size={16} className="ml-0.5" />}
      </button>
      <button onClick={() => setFrame(Math.max(0, state.currentFrameIndex - 1))} disabled={state.currentFrameIndex === 0} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm text-[#9a9a9a] transition hover:text-white disabled:opacity-30" aria-label="Previous frame"><SkipBack size={15} /></button>
      <button onClick={() => setFrame(Math.min(maxIndex, state.currentFrameIndex + 1))} disabled={state.currentFrameIndex >= maxIndex} className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm text-[#9a9a9a] transition hover:text-white disabled:opacity-30" aria-label="Next frame"><SkipForward size={15} /></button>

      <div className="relative min-w-0 flex-1">
        <Slider value={[state.currentFrameIndex]} min={0} max={maxIndex || 1} step={1} onValueChange={([v]) => { setPlaying(false); setFrame(v); }} disabled={maxIndex === 0} className="[&_[data-radix-slider-track]]:h-1.5 [&_[data-radix-slider-track]]:bg-white/20 [&_[data-radix-slider-range]]:bg-[#b8ff24] [&_[data-radix-slider-thumb]]:h-3.5 [&_[data-radix-slider-thumb]]:w-3.5 [&_[data-radix-slider-thumb]]:border-2 [&_[data-radix-slider-thumb]]:border-[#b8ff24] [&_[data-radix-slider-thumb]]:bg-black [&_[data-radix-slider-thumb]]:shadow-[0_0_10px_rgba(184,255,36,0.35)]" aria-label="Scrub through simulation frames" />
        <div className="mt-0.5 flex justify-between font-tabular text-[9px] text-[#666]">{timelineMarks.map((m) => <span key={m}>{m} min</span>)}</div>
      </div>

      <div className="flex shrink-0 gap-2">{SPEEDS.map((s) => <button key={s} onClick={() => setSpeed(s)} className={cn('px-1 py-1 font-tabular text-[11px] transition-colors', state.playbackSpeed === s ? 'text-white underline underline-offset-4' : 'text-[#666] hover:text-white')}>{s}x</button>)}</div>
      <div className="shrink-0 border-l border-[var(--varuna-border)] pl-3 text-right"><p className="font-tabular text-sm font-semibold leading-tight text-white">T+{minutes} min</p><p className="font-tabular text-[10px] leading-tight text-[#666]">{timestamp}</p></div>
    </div>
  );
}
