import { DEPTH_BANDS } from '@/lib/demoData';
import { cn } from '@/lib/utils';

interface FloodLegendProps {
  compact?: boolean;
  demo?: boolean;
}

export default function FloodLegend({ compact = false, demo = true }: FloodLegendProps) {
  return (
    <div className={cn('rounded-md border border-[var(--varuna-border)] bg-black/75 backdrop-blur-xl', compact ? 'w-44 p-2.5' : 'w-52 p-3')} role="group" aria-label="Flood depth legend">
      <div className="mb-2 flex items-center justify-between">
        <p className="text-[10px] font-semibold text-white">Flood depth</p>
        <span className="text-[9px] text-[#9a9a9a]">
          {demo ? 'Demo' : 'Backend'}
        </span>
      </div>
      <div className="space-y-1">
        {DEPTH_BANDS.map((band) => (
          <div key={band.label} className="group flex items-center gap-2">
            <span className="inline-block h-2.5 w-7 rounded-sm transition-transform group-hover:scale-x-110" style={{ backgroundColor: band.color }} />
            <span className="font-tabular text-[11px] text-[#d9dbe0]">{band.label}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 border-t border-[var(--varuna-border)] pt-1.5 text-[9px] leading-tight text-[#666]">Visualization bands only — not safety thresholds.</p>
    </div>
  );
}
