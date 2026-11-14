import { DEPTH_BANDS } from '@/lib/demoData';
import { cn } from '@/lib/utils';

interface FloodLegendProps {
  compact?: boolean;
  demo?: boolean;
}

export default function FloodLegend({ compact = false, demo = true }: FloodLegendProps) {
  return (
    <div className={cn('rounded-sm border border-[#536653] bg-[#0d1718]/90 shadow-[0_10px_30px_rgba(0,0,0,0.4)] backdrop-blur-xl', compact ? 'w-44 p-2.5' : 'w-52 p-3')} role="group" aria-label="Flood depth legend">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-[#f0f1f3]">Flood intensity</p>
        <span className="rounded-sm bg-[#d7ff3f]/15 px-1 py-0.5 text-[8px] font-semibold uppercase tracking-wider text-[#d7ff3f]">
          {demo ? 'Demo' : 'Backend'}
        </span>
      </div>
      <div className="space-y-1">
        {DEPTH_BANDS.map((band) => (
          <div key={band.label} className="group flex items-center gap-2">
            <span className="inline-block h-3 w-6 rounded-sm transition-transform group-hover:scale-x-110" style={{ backgroundColor: band.color, boxShadow: `0 0 8px ${band.color}44` }} />
            <span className="font-mono-num text-[11px] text-[#d9dbe0]">{band.label}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 border-t border-[#4d505b] pt-1.5 text-[9px] leading-tight text-[#9598a1]">Visualization bands only — not safety thresholds.</p>
    </div>
  );
}
