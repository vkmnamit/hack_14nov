import { AlertCircle, BarChart3, Clock3, Layers3, ShieldAlert } from 'lucide-react';

const METRICS = [
  { label: 'MAE', icon: <BarChart3 size={14} /> },
  { label: 'RMSE', icon: <BarChart3 size={14} /> },
  { label: 'Flood extent IoU', icon: <Layers3 size={14} /> },
  { label: 'Inference time', icon: <Clock3 size={14} /> },
  { label: 'Solver runtime', icon: <Clock3 size={14} /> },
  { label: 'Mass-balance error', icon: <ShieldAlert size={14} /> },
];

export default function ModelPerformancePanel() {
  return (
    <div className="panel-scroll flex h-full flex-col gap-4 overflow-y-auto bg-[#0b1015] p-4">
      <div className="border-b border-[#202b34] pb-3">
        <p className="font-mono-num text-[9px] uppercase tracking-[0.18em] text-[#7e8b96]">
          Evaluation workspace
        </p>
        <h2 className="mt-1 text-sm font-medium text-[#e8edf2]">Model performance</h2>
      </div>

      <div className="border border-[#3b2b25] bg-[#17120f] p-3">
        <div className="flex items-start gap-2">
          <AlertCircle size={15} className="mt-0.5 shrink-0 text-[#ffb547]" />
          <div>
            <p className="text-xs font-medium text-[#ffb547]">Evaluation data unavailable</p>
            <p className="mt-1 text-[10px] leading-relaxed text-[#a88f7e]">
              The current backend contains datasets but no evaluation API or prediction response
              contract. Metrics are intentionally not fabricated.
            </p>
          </div>
        </div>
      </div>

      <section className="border border-[#202b34] bg-[#111820] p-3">
        <p className="mb-3 text-[10px] uppercase tracking-[0.14em] text-[#7e8b96]">
          Required backend fields
        </p>
        <div className="space-y-2">
          {METRICS.map((metric) => (
            <div key={metric.label} className="flex items-center justify-between border-b border-[#202b34] pb-2 last:border-b-0 last:pb-0">
              <span className="flex items-center gap-2 text-[11px] text-[#aeb9c2]">
                <span className="text-[#00cfff]">{metric.icon}</span>
                {metric.label}
              </span>
              <span className="font-mono-num text-[10px] text-[#7e8b96]">N/A</span>
            </div>
          ))}
        </div>
      </section>

      <div className="mt-auto text-[10px] leading-relaxed text-[#65727d]">
        Expected integration: reference depth frames, AI depth frames, error frames, metrics and
        runtimes from a typed evaluation endpoint. No endpoint exists in the current backend.
      </div>
    </div>
  );
}
