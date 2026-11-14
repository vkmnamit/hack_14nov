import { AlertCircle, BarChart3, Clock3, Layers3, ShieldAlert } from 'lucide-react';
import { useSimulation } from '@/state/SimulationContext';

const MODEL_NAMES = {
  physics: 'ANUGA physics-only',
  convlstm: 'ConvLSTM-only',
  adaptive: 'ConvLSTM + SWE fallback',
} as const;

export default function ModelPerformancePanel() {
  const { state } = useSimulation();
  const diagnostics = state.simulationMeta?.diagnostics;
  const operationalMetrics = [
    { label: 'Selected model', icon: <Layers3 size={14} />, value: MODEL_NAMES[state.modelType] },
    { label: 'Active solver', icon: <ShieldAlert size={14} />, value: diagnostics?.solver ?? 'Run a forecast' },
    {
      label: 'Rollout time',
      icon: <Clock3 size={14} />,
      value: diagnostics ? `${diagnostics.runtimeSeconds.toFixed(3)} s` : '—',
    },
    {
      label: 'SWE fallback steps',
      icon: <BarChart3 size={14} />,
      value: diagnostics ? String(diagnostics.physicsFallbackSteps) : '—',
    },
  ];

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
            <p className="text-xs font-medium text-[#ffb547]">Accuracy is not measured for this run</p>
            <p className="mt-1 text-[10px] leading-relaxed text-[#a88f7e]">
              Interactive forecasts do not have a ground-truth flood map. MAE, RMSE and overlap
              scores are shown only when predictions are compared with reference simulations.
            </p>
          </div>
        </div>
      </div>

      <section className="border border-[#202b34] bg-[#111820] p-3">
        <p className="mb-3 text-[10px] uppercase tracking-[0.14em] text-[#7e8b96]">
          Current forecast
        </p>
        <div className="space-y-2">
          {operationalMetrics.map((metric) => (
            <div key={metric.label} className="flex items-center justify-between border-b border-[#202b34] pb-2 last:border-b-0 last:pb-0">
              <span className="flex items-center gap-2 text-[11px] text-[#aeb9c2]">
                <span className="text-[#00cfff]">{metric.icon}</span>
                {metric.label}
              </span>
              <span className="max-w-[155px] truncate text-right font-mono-num text-[10px] text-[#7e8b96]">
                {metric.value}
              </span>
            </div>
          ))}
        </div>
      </section>

      <div className="mt-auto text-[10px] leading-relaxed text-[#65727d]">
        For benchmark scores, run the held-out ANUGA evaluation. Simulation outputs are synthetic
        scenarios, not observed flood measurements.
      </div>
    </div>
  );
}
