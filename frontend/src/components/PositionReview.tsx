import { useMemo, useState } from "react";
import type { Decision, DecisionAction, ReplayState } from "../types";

interface Props {
  state: ReplayState;
  selectedTicker: string;
  onSaveDecision: (body: unknown) => Promise<void>;
  onFinalizeEvening: () => Promise<void>;
  onReviseMorning: (decisionId: string, body: unknown) => Promise<void>;
  onCompleteMorning: () => Promise<void>;
}

interface Draft {
  action: DecisionAction;
  quantity?: number;
  allocation_sek?: number;
  stop_price?: number;
  target_price?: number;
  risk_label?: string;
}

function morningDraft(decision: Decision): Draft {
  return {
    action: decision.action,
    quantity: decision.quantity ?? undefined,
    allocation_sek: decision.allocation_sek ?? undefined,
    stop_price: decision.stop_price ?? undefined,
    target_price: decision.target_price ?? undefined,
  };
}

export function PositionReview({ state, selectedTicker, onSaveDecision, onFinalizeEvening, onReviseMorning, onCompleteMorning }: Props) {
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [morningDrafts, setMorningDrafts] = useState<Record<string, Draft>>({});
  const [buy, setBuy] = useState<Draft>({ action: "buy", allocation_sek: 10_000, risk_label: "standard" });
  const openPositions = state.positions.filter((position) => position.status === "open");
  const pendingByPosition = useMemo(
    () => new Map(state.pending_decisions.map((decision) => [decision.position_id, decision])),
    [state.pending_decisions],
  );

  if (state.session.phase === "completed") {
    return (
      <section className="panel">
        <h2>Replay completed</h2>
        <p>The replay reached its configured end date. Its positions, decisions, fills, and metric history remain available for review.</p>
      </section>
    );
  }

  if (state.session.phase === "morning") {
    return (
      <section className="panel">
        <h2>Morning confirmation</h2>
        <p className="mode-explanation">Only today’s opening prices are revealed. Save any revision, cancel an action, or execute the unchanged evening plan.</p>
        {state.pending_decisions.map((decision) => {
          const draft = morningDrafts[decision.id] ?? morningDraft(decision);
          const set = (changes: Partial<Draft>) => setMorningDrafts({ ...morningDrafts, [decision.id]: { ...draft, ...changes } });
          const existingPosition = decision.position_id != null;
          return (
            <div className="decision-card" key={decision.id}>
              <div><strong>{decision.ticker}</strong><span>Open {state.morning_open_prices[decision.ticker]?.toFixed(2) ?? "—"}</span></div>
              <div className="inline-fields">
                <label>Action<select value={draft.action} onChange={(event) => set({ action: event.target.value as DecisionAction })}>
                  {existingPosition ? <><option value="keep">Keep</option><option value="reduce">Reduce</option><option value="sell">Sell all</option></> : <option value="buy">Buy</option>}
                </select></label>
                {(draft.action === "reduce" || draft.action === "buy") && <label>Quantity<input type="number" min={1} value={draft.quantity ?? ""} onChange={(event) => set({ quantity: event.target.value ? Number(event.target.value) : undefined })} /></label>}
                {draft.action === "buy" && <label>Allocation (SEK)<input type="number" value={draft.allocation_sek ?? ""} onChange={(event) => set({ allocation_sek: event.target.value ? Number(event.target.value) : undefined })} /></label>}
                {(draft.action === "keep" || draft.action === "reduce" || draft.action === "buy") && <>
                  <label>Stop<input type="number" step="any" value={draft.stop_price ?? ""} onChange={(event) => set({ stop_price: event.target.value ? Number(event.target.value) : undefined })} /></label>
                  <label>Target<input type="number" step="any" value={draft.target_price ?? ""} onChange={(event) => set({ target_price: event.target.value ? Number(event.target.value) : undefined })} /></label>
                </>}
                <button onClick={() => onReviseMorning(decision.id, draft)}>Save revision</button>
                <button className="danger-link" onClick={() => onReviseMorning(decision.id, { cancelled: true })}>Cancel action</button>
              </div>
            </div>
          );
        })}
        <button className="primary" onClick={onCompleteMorning}>Execute confirmed actions and reveal the full day</button>
      </section>
    );
  }

  return (
    <section className="panel">
      <h2>Evening position review</h2>
      {openPositions.length === 0 && <p className="muted">No open positions.</p>}
      {openPositions.map((position) => {
        const existing = pendingByPosition.get(position.id);
        const draft = drafts[position.id] ?? {
          action: existing?.action ?? "keep",
          quantity: existing?.quantity ?? undefined,
          stop_price: existing?.stop_price ?? position.stop_price ?? undefined,
          target_price: existing?.target_price ?? position.target_price ?? undefined,
        };
        const set = (changes: Partial<Draft>) => setDrafts({ ...drafts, [position.id]: { ...draft, ...changes } });
        return (
          <div className="decision-card" key={position.id}>
            <div className="position-summary"><strong>{position.ticker}</strong><span>{position.quantity} shares</span><span>Entry {position.entry_price.toFixed(2)}</span></div>
            <div className="inline-fields">
              <label>Decision<select value={draft.action} onChange={(event) => set({ action: event.target.value as DecisionAction })}><option value="keep">Keep</option><option value="reduce">Reduce on next open</option><option value="sell">Sell on next open</option></select></label>
              {draft.action === "reduce" && <label>Shares to sell<input type="number" min={1} max={position.quantity - 1} value={draft.quantity ?? ""} onChange={(event) => set({ quantity: Number(event.target.value) })} /></label>}
              {draft.action !== "sell" && <><label>New stop<input type="number" step="any" value={draft.stop_price ?? ""} onChange={(event) => set({ stop_price: Number(event.target.value) })} /></label><label>New target<input type="number" step="any" value={draft.target_price ?? ""} onChange={(event) => set({ target_price: Number(event.target.value) })} /></label></>}
              <button onClick={() => onSaveDecision({ ...draft, ticker: position.ticker, position_id: position.id })}>Save decision</button>
            </div>
          </div>
        );
      })}

      <h3>Prepare new entry: {selectedTicker}</h3>
      <div className="inline-fields">
        <label>Capital allocation<input type="number" value={buy.allocation_sek ?? ""} onChange={(event) => setBuy({ ...buy, allocation_sek: Number(event.target.value) })} /></label>
        <label>Risk level<select value={buy.risk_label} onChange={(event) => setBuy({ ...buy, risk_label: event.target.value })}><option value="tight">Tight</option><option value="standard">Standard</option><option value="wide">Wide</option><option value="custom">Custom</option></select></label>
        <label>Stop price<input type="number" step="any" value={buy.stop_price ?? ""} onChange={(event) => setBuy({ ...buy, stop_price: Number(event.target.value) })} /></label>
        <label>Target price<input type="number" step="any" value={buy.target_price ?? ""} onChange={(event) => setBuy({ ...buy, target_price: Number(event.target.value) })} /></label>
        <button onClick={() => onSaveDecision({ ...buy, ticker: selectedTicker })}>Prepare buy</button>
      </div>
      <button className="primary" disabled={state.outstanding_position_ids.length > 0} onClick={onFinalizeEvening}>
        {state.outstanding_position_ids.length > 0 ? `${state.outstanding_position_ids.length} position decisions remaining` : "Finalize decisions and reveal next-day opens"}
      </button>
    </section>
  );
}
