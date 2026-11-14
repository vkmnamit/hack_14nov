import { Building2, Droplet, Download, Route, Ruler, Timer, Waves } from 'lucide-react';
import { BarChart, Bar, Cell, Tooltip as ReTooltip, XAxis, YAxis } from 'recharts';
import MetricCard from '@/components/dashboard/MetricCard';
import PhysicsIntegrity from '@/components/dashboard/PhysicsIntegrity';
import { useSimulation } from '@/state/SimulationContext';
import { computeStats } from '@/lib/demoData';
import { apiConfig } from '@/services/api';

const BAND_COLORS = ['#9BD8F5', '#00D4FF', '#1560D4', '#FFB547', '#FF4545'];

export default function AnalyticsPanel() {
  const { state } = useSimulation();
  const frame = state.dataset?.frames[state.currentFrameIndex] ?? null;
  const exposure = state.dataset?.exposure;
  const stats = frame && state.dataset
    ? computeStats(frame, state.dataset.terrain, state.isDemoMode ? 600 : 0, state.isDemoMode ? 140 : 0)
    : null;
  const exposureValue = (
    count: number | undefined,
    demoCount: number | undefined
  ): string => {
    if (state.isDemoMode) return demoCount === undefined ? '—' : String(demoCount);
    if (count !== undefined) return String(count);
    if (exposure?.status === 'unavailable') return 'OSM unavailable';
    return state.simulatedOnce ? 'OSM pending' : '—';
  };

  const chartData =
    stats?.severityCounts.map((s, i) => ({ name: s.band, count: s.count, fill: BAND_COLORS[i] })) ??
    [];

  return (
    <div className="panel-scroll flex h-full flex-col gap-3 overflow-y-auto bg-black/70 p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold text-white">
          Forecast analytics
        </h2>
        <span className="border border-[var(--varuna-border)] px-2 py-1 text-[9px] text-[#9a9a9a]">
          {state.isDemoMode ? 'Illustrative' : 'Backend'}
        </span>
      </div>

      {state.simulationMeta?.id && !state.isDemoMode && (
        <div className="grid gap-2">
          <a
            href={`${apiConfig.baseUrl}/api/simulation/${state.simulationMeta.id}/report`}
            download={`${state.simulationMeta.id}-risk-report.png`}
            className="flex items-center justify-center gap-2 rounded border border-[#253540] bg-[#0b1418] px-3 py-2 text-[11px] text-[#b8ff24] transition hover:bg-[#b8ff24]/10"
          >
            <Download size={13} /> Download depth &amp; risk map
          </a>
          <a
            href={`${apiConfig.baseUrl}/api/simulation/${state.simulationMeta.id}/comparison-report`}
            download={`${state.simulationMeta.id}-model-comparison.html`}
            className="flex items-center justify-center gap-2 rounded border border-[#253540] bg-[#0b1418] px-3 py-2 text-[11px] text-[#b8ff24] transition hover:bg-[#b8ff24]/10"
          >
            <Download size={13} /> All-model maps + Ollama verdict
          </a>
        </div>
      )}

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-2">
        <MetricCard
          label="Max Water Depth"
          value={stats ? stats.maxDepth.toFixed(3) : '—'}
          unit="m"
          icon={<Waves size={16} />}
          accent="cyan"
          tooltip={`Deepest simulated water column across the ${state.dataset?.terrain.size ?? '—'} × ${state.dataset?.terrain.size ?? '—'} grid at the current time step.`}
        />
        <MetricCard
          label="Flooded Area"
          value={stats ? stats.floodedAreaKm2.toFixed(3) : '—'}
          unit="km²"
          icon={<Droplet size={16} />}
          accent="blue"
          tooltip={`Total area with water depth ≥ 1 cm, computed from ${(state.dataset?.terrain.cellSizeM ?? 25)} m grid cells.`}
        />
        <MetricCard
          label="Affected Buildings"
          value={exposureValue(frame?.affectedBuildings, stats?.affectedBuildings)}
          icon={<Building2 size={16} />}
          accent="amber"
          tooltip={state.isDemoMode ? 'Illustrative demo count; not an OSM inventory.' : exposure?.method ?? 'Building centroids in cells with depth ≥ 1 cm, counted from OpenStreetMap.'}
        />
        <MetricCard
          label="Affected Road Ways"
          value={exposureValue(frame?.affectedRoadWays, stats?.affectedRoadSegments)}
          icon={<Route size={16} />}
          accent="amber"
          tooltip={state.isDemoMode ? 'Illustrative demo count; not an OSM inventory.' : exposure?.method ?? 'OpenStreetMap road ways crossing cells with depth ≥ 1 cm.'}
        />
        <MetricCard
          label="Simulation Time"
          value={frame ? `${frame.t}` : '—'}
          unit="min"
          icon={<Timer size={16} />}
          accent="neutral"
          tooltip="Elapsed simulation time of the active frame."
        />
      </div>

      {/* Depth statistics */}
      <section className="rounded-md border border-[var(--varuna-border)] bg-black/60 p-3">
        <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold text-white">
          <Ruler size={12} className="text-[var(--varuna-blue-2)]" /> Depth statistics
        </h3>
        <div className="grid grid-cols-3 gap-2 font-tabular text-center">
          <div>
            <p className="text-[10px] text-slate-500">Max</p>
            <p className="text-sm font-semibold text-[var(--varuna-blue-2)]">
              {stats ? `${(stats.maxDepth * 100).toFixed(1)}` : '—'} <span className="text-[10px] text-slate-500">cm</span>
            </p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">Mean*</p>
            <p className="text-sm font-semibold text-slate-200">
              {stats ? `${(stats.meanDepth * 100).toFixed(1)}` : '—'} <span className="text-[10px] text-slate-500">cm</span>
            </p>
          </div>
          <div>
            <p className="text-[10px] text-slate-500">Wet cells</p>
            <p className="text-sm font-semibold text-slate-200">
              {stats ? (stats.floodedFraction * 100).toFixed(0) : '—'} <span className="text-[10px] text-slate-500">%</span>
            </p>
          </div>
        </div>
        <p className="mt-2 text-[9px] leading-tight text-slate-600">
          *Mean over all cells, wet and dry. {state.isDemoMode ? 'Synthetic demo data.' : 'Backend forecast data.'}
        </p>

        {/* Severity distribution */}
        <h3 className="mb-2 mt-4 text-[11px] font-semibold text-white">
          Severity distribution
        </h3>
        <div className="h-32">
          {chartData.length > 0 ? (
            <BarChart width={280} height={128} data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -18 }}>
                <XAxis
                  dataKey="name"
                  tick={{ fill: '#64748b', fontSize: 9 }}
                  axisLine={{ stroke: '#1b2840' }}
                  tickLine={false}
                />
                <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={false} tickLine={false} />
                <ReTooltip
                  cursor={{ fill: 'rgba(0, 212, 255, 0.06)' }}
                  contentStyle={{
                    background: '#101827',
                    border: '1px solid #1b2840',
                    borderRadius: 6,
                    fontSize: 11,
                    color: '#e6edf5',
                  }}
                  formatter={(value: number | string) => [`${value} cells`, 'Cells']}
                />
                <Bar dataKey="count" radius={[3, 3, 0, 0]}>
                  {chartData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
            </BarChart>
          ) : (
            <div className="flex h-full items-center justify-center px-4 text-center text-xs text-slate-600">
              {state.status === 'running' ? 'Simulation in progress. Severity distribution will appear with the first frame.' : 'Run a simulation to populate depth statistics, severity counts, and risk metrics for this location.'}
            </div>
          )}
        </div>
      </section>

      {/* Physics integrity */}
      <PhysicsIntegrity />
    </div>
  );
}
