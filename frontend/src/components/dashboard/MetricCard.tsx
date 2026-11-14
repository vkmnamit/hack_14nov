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

const ACCENTS: Record<NonNullable<MetricCardProps['accent']>, { text: string; rail: string; glow: string }> = {
  cyan: { text: 'text-[#80e8ff]', rail: 'bg-[#80e8ff]', glow: 'shadow-[0_0_12px_rgba(128,232,255,0.25)]' },
  blue: { text: 'text-[#7ab0e8]', rail: 'bg-[#7ab0e8]', glow: 'shadow-[0_0_12px_rgba(122,176,232,0.2)]' },
  amber: { text: 'text-[#d7ff3f]', rail: 'bg-[#d7ff3f]', glow: 'shadow-[0_0_12px_rgba(215,255,63,0.25)]' },
  red: { text: 'text-[#ed7044]', rail: 'bg-[#ed7044]', glow: 'shadow-[0_0_12px_rgba(237,112,68,0.25)]' },
  neutral: { text: 'text-[#d9dbe0]', rail: 'bg-[#8ea79b]', glow: 'shadow-[0_0_12px_rgba(142,167,155,0.16)]' },
};

export default function MetricCard({ label, value, unit, icon, tooltip, accent = 'cyan' }: MetricCardProps) {
  const a = ACCENTS[accent];
  return (
    <Card className="group relative overflow-hidden rounded-xl border-[#26332c] bg-gradient-to-br from-[#151c19]/95 to-[#090c0d]/95 shadow-[0_8px_24px_rgba(0,0,0,0.28)] backdrop-blur transition-all duration-200 hover:border-[#789d3c] hover:shadow-[0_0_18px_rgba(184,255,36,0.12)]" title={tooltip}>
      <div className={cn('absolute left-0 top-0 h-full w-0.5', a.rail, a.glow)} />
      <CardContent className="flex items-center gap-3 px-3 py-2.5 pl-4">
        <div className={cn('shrink-0 rounded-lg bg-[#1d2a20] p-2 transition-transform duration-200 group-hover:scale-105', a.text)}>
          {icon}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-[10px] font-medium uppercase tracking-wider text-[#aeb1b8]">{label}</p>
          <p className="font-mono-num text-lg font-semibold leading-tight text-[#f0f1f3]">
            {value}
            {unit && <span className="ml-1 text-xs font-normal text-[#aeb1b8]">{unit}</span>}
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
