import type { MetricSnapshot, PositionScatterPoint } from "../types";

function formatPercent(value: number | null): string {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function MetricLine({ history, field }: { history: MetricSnapshot[]; field: keyof MetricSnapshot }) {
  const values = history.map((item) => Number(item[field])).filter(Number.isFinite);
  if (values.length < 2) return <div className="mini-chart empty">Insufficient history</div>;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const points = values.map((value, index) => `${(index / (values.length - 1)) * 100},${38 - ((value - min) / range) * 34}`).join(" ");
  return <svg viewBox="0 0 100 40" className="mini-chart"><polyline points={points} fill="none" stroke="currentColor" strokeWidth="1.8" /></svg>;
}

function PositionScatter({ points }: { points: PositionScatterPoint[] }) {
  if (points.length === 0) return <div className="scatter empty">No open positions</div>;
  const maxDays = Math.max(...points.map((point) => point.days_owned), 1);
  const maxAbs = Math.max(...points.map((point) => Math.abs(point.annualized_simple_return)), 0.01);
  return (
    <svg viewBox="0 0 100 60" className="scatter" aria-label="Annualized simple return by days owned">
      <line x1="5" x2="98" y1="30" y2="30" stroke="currentColor" opacity="0.25" />
      {points.map((point) => {
        const x = 5 + (point.days_owned / maxDays) * 90;
        const y = 30 - (point.annualized_simple_return / maxAbs) * 25;
        return <g key={point.ticker}><circle cx={x} cy={y} r="2.2" /><text x={x + 2.5} y={y - 1} fontSize="3.5">{point.ticker}</text></g>;
      })}
      <text x="50" y="59" textAnchor="middle" fontSize="4">Trading sessions owned</text>
    </svg>
  );
}

export function Metrics({ metrics, history, scatter }: { metrics: MetricSnapshot | null; history: MetricSnapshot[]; scatter: PositionScatterPoint[] }) {
  const cards = [
    ["Total return", formatPercent(metrics?.total_return ?? null), "total_return"],
    ["Expectancy", metrics?.expectancy_r == null ? "—" : `${metrics.expectancy_r.toFixed(2)}R`, "expectancy_r"],
    ["Win rate", formatPercent(metrics?.win_rate ?? null), "win_rate"],
    ["Cumulative R", metrics?.cumulative_r == null ? "—" : `${metrics.cumulative_r.toFixed(2)}R`, "cumulative_r"],
    ["Sharpe", metrics?.sharpe_ratio == null ? "—" : metrics.sharpe_ratio.toFixed(2), "sharpe_ratio"],
    ["Sortino", metrics?.sortino_ratio == null ? "—" : metrics.sortino_ratio.toFixed(2), "sortino_ratio"],
  ] as const;
  return (
    <section className="panel metrics-panel">
      <h2>Replay performance</h2>
      <div className="metric-grid">
        {cards.map(([label, value, field]) => <article key={label} className="metric-card"><span>{label}</span><strong>{value}</strong><MetricLine history={history} field={field} /></article>)}
      </div>
      <h3>Current positions: annualized simple return vs. days owned</h3>
      <PositionScatter points={scatter} />
    </section>
  );
}
