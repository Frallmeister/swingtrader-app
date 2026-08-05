import type { ReplayEvent, ReplayFill } from "../types";

export function AuditTrail({ fills, events }: { fills: ReplayFill[]; events: ReplayEvent[] }) {
  return (
    <section className="panel">
      <h2>Execution and decision history</h2>
      <details open>
        <summary>Fills ({fills.length})</summary>
        <div className="audit-list">
          {fills.slice().reverse().slice(0, 30).map((fill) => (
            <div key={fill.id} className="audit-row">
              <span>{fill.trading_date}</span><strong>{fill.ticker}</strong><span>{fill.side} {fill.quantity} @ {fill.price.toFixed(2)}</span><span>{fill.courtage.toFixed(2)} SEK courtage</span>
            </div>
          ))}
        </div>
      </details>
      <details>
        <summary>Decision log ({events.length})</summary>
        <div className="audit-list">
          {events.map((event) => (
            <div key={event.id} className="audit-row">
              <span>#{event.sequence_number}</span><span>{event.trading_date}</span><strong>{event.event_type.replaceAll("_", " ")}</strong><span>{event.ticker ?? "Replay"}</span>
            </div>
          ))}
        </div>
      </details>
    </section>
  );
}
