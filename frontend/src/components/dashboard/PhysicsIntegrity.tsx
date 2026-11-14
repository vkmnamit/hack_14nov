import { Info, ShieldCheck } from 'lucide-react';
import { demoPhysicsChecks, type PhysicsCheck } from '@/lib/demoData';
import { useSimulation } from '@/state/SimulationContext';
import { cn } from '@/lib/utils';

const CHECK_STYLES: Record<PhysicsCheck['state'], { dot: string; text: string; label: string }> = {
  pass: { dot: 'bg-[#34d399]', text: 'text-[#34d399]', label: 'PASS' },
  warning: { dot: 'bg-[#FFB547]', text: 'text-[#FFB547]', label: 'WARNING' },
  unavailable: { dot: 'bg-slate-600', text: 'text-slate-500', label: 'UNAVAILABLE' },
};

export default function PhysicsIntegrity() {
  const { state } = useSimulation();
  const frame = state.dataset?.frames[state.currentFrameIndex] ?? null;
  const prevFrame =
    state.dataset && state.currentFrameIndex > 0
      ? state.dataset.frames[state.currentFrameIndex - 1]
      : null;
  const checks = frame ? demoPhysicsChecks(frame, prevFrame) : null;

  return (
    <section className="rounded-md border border-[var(--varuna-border)] bg-black/60 p-3">
      <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold text-white">
        <ShieldCheck size={12} className="text-[var(--varuna-blue-2)]" /> Physics integrity
      </h3>
      <div className="space-y-1.5">
        {(checks ?? [
          {
            id: 'idle',
            label: 'Awaiting simulation',
            description: '',
            state: 'unavailable' as const,
            detail: 'Run a simulation to evaluate diagnostics.',
          },
        ]).map((c: PhysicsCheck) => {
          const s = CHECK_STYLES[c.state];
          return (
            <div
              key={c.id}
              className="flex items-start justify-between gap-2 rounded-md border border-[var(--varuna-border)] bg-black/50 px-2.5 py-2"
            >
              <div className="min-w-0">
                <p className="truncate text-[11px] font-medium text-slate-200" title={c.description}>
                  {c.label}
                </p>
                <p className="mt-0.5 text-[9px] leading-tight text-slate-500">{c.detail}</p>
              </div>
              <div className="flex shrink-0 items-center gap-1.5">
                <span className={cn('h-1.5 w-1.5 rounded-full', s.dot)} />
                <span className={cn('font-mono-num text-[9px] font-semibold tracking-wider', s.text)}>
                  {s.label}
                </span>
              </div>
            </div>
          );
        })}
      </div>
      <p className="mt-2 flex items-start gap-1 text-[9px] leading-tight text-slate-600">
        <Info size={10} className="mt-0.5 shrink-0" />
        Demo diagnostics are illustrative. Real checks require the FastAPI solver backend.
      </p>
    </section>
  );
}
