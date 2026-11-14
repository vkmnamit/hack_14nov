import { useSimulation } from '@/state/SimulationContext';
import { computeStats } from '@/lib/demoData';

function SummaryItem({
  label,
  value,
  unit,
  accent = 'text-slate-100',
}: {
  label: string;
  value: string;
  unit?: string;
  accent?: string;
}) {
  return (
    <div className="min-w-0 flex-1 border-r border-[var(--varuna-border)] px-4 last:border-r-0">
      <div className="mb-1 text-[10px] font-medium text-[#9a9a9a]">
        {label}
      </div>
      <p className={`font-tabular text-[20px] font-medium leading-none tracking-tight ${accent}`}>
        {value}
        {unit && <span className="ml-1 text-[10px] font-normal text-[#aeb1b8]">{unit}</span>}
      </p>
    </div>
  );
}

export default function OperationsSummary() {
  const { state } = useSimulation();
  const frame = state.dataset?.frames[state.currentFrameIndex] ?? null;
  const stats = frame && state.dataset ? computeStats(frame, state.dataset.terrain, 600, 140) : null;
  const statusAccent =
    state.status === 'completed'
      ? 'text-white'
      : state.status === 'running'
        ? 'text-[#f4ca35]'
        : state.status === 'error'
          ? 'text-[#ff3939]'
          : 'text-[#dfe1e5]';

  return (
    <section className="flex min-h-[74px] shrink-0 items-center border-b border-[var(--varuna-border)] bg-black/70 px-1 lg:px-3">
      <SummaryItem label="Rainfall intensity" value={`${state.rainfallIntensity}`} unit="mm/hr" accent="text-white" />
      <SummaryItem label="Operational status" value={state.status.toUpperCase()} accent={statusAccent} />
      <SummaryItem label="Max water depth" value={stats ? stats.maxDepth.toFixed(2) : '—'} unit="m" accent="text-[#65c9e7]" />
      <SummaryItem label="Flooded area" value={stats ? stats.floodedAreaKm2.toFixed(3) : '—'} unit="km²" accent="text-[#e5e7eb]" />
      <SummaryItem label="Buildings at risk" value={stats ? `${stats.affectedBuildings}` : '—'} accent="text-[#f4ca35]" />
      <SummaryItem label="Road segments" value={stats ? `${stats.affectedRoadSegments}` : '—'} accent="text-[#ed7044]" />
      <div className="hidden min-w-[150px] flex-1 border-r border-[var(--varuna-border)] px-4 xl:block">
        <div>
          <p className="text-[10px] text-[#9a9a9a]">Model integrity</p>
          <p className="font-tabular text-sm text-white">Demo / illustrative</p>
        </div>
      </div>
    </section>
  );
}
