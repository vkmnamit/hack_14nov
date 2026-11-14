import { Activity, Building2, CloudRain, Droplets, Route, ShieldCheck, Waves } from 'lucide-react';
import { useSimulation } from '@/state/SimulationContext';
import { computeStats } from '@/lib/demoData';

function SummaryItem({
  label,
  value,
  unit,
  accent = 'text-slate-100',
  icon,
}: {
  label: string;
  value: string;
  unit?: string;
  accent?: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="min-w-[142px] flex-1 rounded-xl border border-[#26332c] bg-gradient-to-br from-[#121917] to-[#090c0d] px-4 py-3 shadow-[0_8px_24px_rgba(0,0,0,0.28)]">
      <div className="mb-1 flex items-center gap-1.5 text-[10px] font-medium uppercase tracking-[0.14em] text-[#82958a]">
        <span className={accent}>{icon}</span>
        {label}
      </div>
      <p className={`font-mono-num text-[21px] font-medium leading-none tracking-tight ${accent}`}>
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
      ? 'text-[#9ad52d]'
      : state.status === 'running'
        ? 'text-[#f4ca35]'
        : state.status === 'error'
          ? 'text-[#ff3939]'
          : 'text-[#dfe1e5]';

  return (
    <section className="flex min-h-[86px] shrink-0 flex-wrap items-center gap-2 border-b border-[#101816] bg-[#060909]/80 px-3 py-3 shadow-[0_10px_30px_rgba(0,0,0,0.22)] lg:px-4">
      <SummaryItem label="Rainfall intensity" value={`${state.rainfallIntensity}`} unit="mm/hr" accent="text-[#9ad52d]" icon={<CloudRain size={13} />} />
      <SummaryItem label="Operational status" value={state.status.toUpperCase()} accent={statusAccent} icon={<Activity size={13} />} />
      <SummaryItem label="Max water depth" value={stats ? stats.maxDepth.toFixed(2) : '—'} unit="m" accent="text-[#65c9e7]" icon={<Waves size={13} />} />
      <SummaryItem label="Flooded area" value={stats ? stats.floodedAreaKm2.toFixed(3) : '—'} unit="km²" accent="text-[#e5e7eb]" icon={<Droplets size={13} />} />
      <SummaryItem label="Buildings at risk" value={stats ? `${stats.affectedBuildings}` : '—'} accent="text-[#f4ca35]" icon={<Building2 size={13} />} />
      <SummaryItem label="Road segments" value={stats ? `${stats.affectedRoadSegments}` : '—'} accent="text-[#ed7044]" icon={<Route size={13} />} />
      <div className="hidden min-w-[178px] flex-1 items-center gap-2 rounded-xl border border-[#26332c] bg-gradient-to-br from-[#121917] to-[#090c0d] px-4 py-3 shadow-[0_8px_24px_rgba(0,0,0,0.28)] xl:flex">
        <div className="flex h-8 w-8 items-center justify-center rounded-full border border-[#536653] text-[#d7ff3f]">
          <ShieldCheck size={15} />
        </div>
        <div>
          <p className="text-[10px] uppercase tracking-[0.12em] text-[#aeb1b8]">Model integrity</p>
          <p className="font-mono-num text-xs text-[#dfe1e5]">DEMO / ILLUSTRATIVE</p>
        </div>
      </div>
    </section>
  );
}
