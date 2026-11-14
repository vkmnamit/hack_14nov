import { Building2, Droplet, Route, Ruler, Timer, Waves } from 'lucide-react';
import { BarChart, Bar, Cell, Tooltip as ReTooltip, XAxis, YAxis } from 'recharts';
import MetricCard from '@/components/dashboard/MetricCard';
import PhysicsIntegrity from '@/components/dashboard/PhysicsIntegrity';
import { useSimulation } from '@/state/SimulationContext';
import { computeStats } from '@/lib/demoData';

const BAND_COLORS = ['#9BD8F5', '#00D4FF', '#1560D4', '#FFB547', '#FF4545'];

export default function AnalyticsPanel() {
  const { state } = useSimulation();
  const frame = state.dataset?.frames[state.currentFrameIndex] ?? null;
  const stats = frame && state.dataset
    ? computeStats(frame, state.dataset.terrain, state.isDemoMode ? 600 : 0, state.isDemoMode ? 140 : 0)
    : null;

  const chartData =
    stats?.severityCounts.map((s, i) => ({ name: s.band, count: s.count, fill: BAND_COLORS[i] })) ??
    [];

  return (
    <div className="panel-scroll flex h-full flex-col gap-3 overflow-y-auto bg-[#080b0c] p-3">
      <div className="flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-widest text-slate-300">
          Forecast Analytics
        </h2>
        <span className="rounded-lg border border-[#FFB547]/40 bg-[#FFB547]/10 px-2 py-1 text-[9px] font-semibold uppercase tracking-wider text-[#FFB547]">
          {state.isDemoMode ? 'Illustrative' : 'Backend'}
        </span>
      </div>

      {/* Metric cards */}
      <div className="grid grid-cols-1 gap-2">
        <MetricCard
          label="Max Water Depth"
          value={stats ? stats.maxDepth.toFixed(3) : '—'}
          unit="m"
          icon={<Waves size={16} />}
          accent="cyan"
          tooltip="Deepest simulated water column across the 64×64 grid at the current time step."
        />
        <MetricCard
          label="Flooded Area"
          value={stats ? stats.floodedAreaKm2.toFixed(3) : '—'}
          unit="km²"
          icon={<Droplet size={16} />}
          accent="blue"
          tooltip="Total area with water depth ≥ 1 cm, computed from the 25 m grid cells."
        />
        <MetricCard
          label="Affected Buildings"
          value={stats && state.isDemoMode ? stats.affectedBuildings.toString() : 'N/A'}
          icon={<Building2 size={16} />}
          accent="amber"
          tooltip={state.isDemoMode ? 'Buildings intersecting any wet cell (synthetic exposure catalogue, demo).' : 'Backend building exposure data is not available.'}
        />
        <MetricCard
          label="Affected Road Segments"
          value={stats && state.isDemoMode ? stats.affectedRoadSegments.toString() : 'N/A'}
          icon={<Route size={16} />}
          accent="amber"
          tooltip={state.isDemoMode ? 'Road segments crossing any wet cell (synthetic network, demo).' : 'Backend road exposure data is not available.'}
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
      <section className="rounded-sm border border-[#263839] bg-[#111c1d] p-3 shadow-[0_8px_24px_rgba(0,0,0,0.18)]">
        <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-widest text-slate-300">
          <Ruler size={12} className="text-[#00D4FF]" /> Depth Statistics
        </h3>
        <div className="grid grid-cols-3 gap-2 font-mono-num text-center">
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Max</p>
            <p className="text-sm font-semibold text-[#00D4FF]">
              {stats ? `${(stats.maxDepth * 100).toFixed(1)}` : '—'} <span className="text-[10px] text-slate-500">cm</span>
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Mean*</p>
            <p className="text-sm font-semibold text-slate-200">
              {stats ? `${(stats.meanDepth * 100).toFixed(1)}` : '—'} <span className="text-[10px] text-slate-500">cm</span>
            </p>
          </div>
          <div>
            <p className="text-[10px] uppercase tracking-wider text-slate-500">Wet Cells</p>
            <p className="text-sm font-semibold text-slate-200">
              {stats ? (stats.floodedFraction * 100).toFixed(0) : '—'} <span className="text-[10px] text-slate-500">%</span>
            </p>
          </div>
        </div>
        <p className="mt-2 text-[9px] leading-tight text-slate-600">
          *Mean over all cells, wet and dry. {state.isDemoMode ? 'Demo data.' : 'Backend frame data.'}
        </p>

        {/* Severity distribution */}
        <h3 className="mb-2 mt-4 text-[11px] font-semibold uppercase tracking-widest text-slate-300">
          Severity Distribution
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
            <div className="flex h-full items-center justify-center text-xs text-slate-600">
              Run a simulation to see the distribution.
            </div>
          )}
        </div>
      </section>

      {/* Physics integrity */}
      <PhysicsIntegrity />
    </div>
  );
}
