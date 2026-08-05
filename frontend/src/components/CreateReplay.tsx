import { useState } from "react";

interface Props {
  onCreate: (body: unknown) => Promise<void>;
  onCancel?: () => void;
}

export function CreateReplay({ onCreate, onCancel }: Props) {
  const [name, setName] = useState("Indicator replay");
  const [tickers, setTickers] = useState("VOLV-B.ST, SAAB-B.ST");
  const [startDate, setStartDate] = useState("2020-01-02");
  const [endDate, setEndDate] = useState("2020-12-30");
  const [cash, setCash] = useState(50_000);
  const [courtage, setCourtage] = useState("mini");
  const [barrier, setBarrier] = useState("stop_first");
  return (
    <main className="create-page"><section className="panel create-card">
      <h1>Create discretionary replay</h1>
      <label>Name<input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <label>Yahoo Finance tickers<textarea value={tickers} onChange={(event) => setTickers(event.target.value)} /></label>
      <div className="inline-fields"><label>Start date<input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></label><label>End date<input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></label></div>
      <label>Initial cash (SEK)<input type="number" value={cash} onChange={(event) => setCash(Number(event.target.value))} /></label>
      <label>Avanza courtage profile<select value={courtage} onChange={(event) => setCourtage(event.target.value)}><option value="mini">Mini</option><option value="small">Small</option><option value="medium">Medium</option><option value="fixed">Fixed</option></select></label>
      <label>Ambiguous same-candle policy<select value={barrier} onChange={(event) => setBarrier(event.target.value)}><option value="stop_first">Stop first</option><option value="target_first">Target first</option><option value="candle_path">Triple-barrier candle path</option></select></label>
      <div className="form-actions">
        {onCancel && <button onClick={onCancel}>Cancel</button>}
        <button className="primary" onClick={() => onCreate({ name, provider: "yfinance", tickers: tickers.split(/[\s,]+/).filter(Boolean), start_date: startDate, end_date: endDate, initial_cash: cash, courtage_profile: courtage, barrier_policy: barrier })}>Create replay</button>
      </div>
    </section></main>
  );
}
