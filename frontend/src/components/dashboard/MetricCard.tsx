import { Card, CardContent } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string;
  unit?: string;
  icon: React.ReactNode;
  tooltip?: string;
  accent?: 'cyan' | 'blue' | 'amber' | 'red' | 'neutral';
}

const ACCENTS: Record<NonNullable<MetricCardProps['accent']>, { text: string; rail: string }> = {
  cyan: { text: 'text-[var(--varuna-blue-2)]', rail: 'bg-[var(--varuna-blue-2)]' },
  blue: { text: 'text-[var(--varuna-blue-2)]', rail: 'bg-[var(--varuna-blue-2)]' },
  amber: { text: 'text-[var(--varuna-amber)]', rail: 'bg-[var(--varuna-amber)]' },
  red: { text: 'text-[#ff6b6b]', rail: 'bg-[#ff6b6b]' },
  neutral: { text: 'text-white', rail: 'bg-white/40' },
};

export default function MetricCard({ label, value, unit, tooltip, accent = 'cyan' }: MetricCardProps) {
  const a = ACCENTS[accent];
  return (
    <Card className="relative overflow-hidden rounded-md border-[var(--varuna-border)] bg-black/70 backdrop-blur" title={tooltip}>
      <div className={cn('absolute left-0 top-0 h-full w-0.5', a.rail)} />
      <CardContent className="flex items-center gap-3 px-3 py-2.5 pl-4">
        <div className="min-w-0 flex-1">
          <p className="truncate text-[10px] font-medium text-[#9a9a9a]">{label}</p>
          <p className={cn('font-tabular text-lg font-semibold leading-tight text-white', a.text)}>
            {value}
            {unit && <span className="ml-1 text-xs font-normal text-[#aeb1b8]">{unit}</span>}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
