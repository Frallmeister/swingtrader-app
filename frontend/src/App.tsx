import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { AuditTrail } from "./components/AuditTrail";
import { CreateReplay } from "./components/CreateReplay";
import { IndicatorPanel } from "./components/IndicatorPanel";
import { MarketChart } from "./components/MarketChart";
import { Metrics } from "./components/Metrics";
import { PositionReview } from "./components/PositionReview";
import { Screener } from "./components/Screener";
import type {
  ChartResponse,
  IndicatorDefinition,
  IndicatorInstance,
  ReplayState,
  ScreeningConfiguration,
  ScreeningPreset,
} from "./types";

export default function App() {
  const [definitions, setDefinitions] = useState<IndicatorDefinition[]>([]);
  const [replays, setReplays] = useState<Array<ReplayState["session"]>>([]);
  const [state, setState] = useState<ReplayState | null>(null);
  const [presets, setPresets] = useState<ScreeningPreset[]>([]);
  const [selectedTicker, setSelectedTicker] = useState("");
  const [instances, setInstances] = useState<IndicatorInstance[]>([]);
  const [chart, setChart] = useState<ChartResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [showCreate, setShowCreate] = useState(false);

  const execute = useCallback(async <T,>(operation: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true);
    setError(null);
    try {
      return await operation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
      return undefined;
    } finally {
      setBusy(false);
    }
  }, []);

  const refreshLists = useCallback(async () => {
    const [indicatorData, replayData, presetData] = await Promise.all([
      api.indicators(), api.replays(), api.presets(),
    ]);
    setDefinitions(indicatorData);
    setReplays(replayData);
    setPresets(presetData);
  }, []);

  useEffect(() => {
    void execute(refreshLists);
  }, [execute, refreshLists]);

  const loadReplay = useCallback(async (id: string) => {
    const loaded = await execute(() => api.replay(id));
    if (!loaded) return;
    setState(loaded);
    setShowCreate(false);
    setSelectedTicker((current) => loaded.session.tickers.includes(current) ? current : loaded.session.tickers[0] || "");
  }, [execute]);

  useEffect(() => {
    if (!state && replays.length > 0) void loadReplay(replays[0].id);
  }, [loadReplay, replays, state]);

  const loadChart = useCallback(async () => {
    if (!state || !selectedTicker) return;
    const response = await execute(() => api.chart(state.session.id, selectedTicker, {
      indicators: instances.map(({ indicator_id, parameters, source }) => ({ indicator_id, parameters, source })),
      lookback_sessions: 240,
    }));
    if (response) setChart(response);
  }, [execute, instances, selectedTicker, state]);

  useEffect(() => {
    void loadChart();
  }, [loadChart]);

  const acceptState = (next: ReplayState | undefined) => {
    if (next) setState(next);
  };

  const phaseTitle = useMemo(() => {
    if (!state) return "";
    if (state.session.phase === "evening") return `Evening review — ${state.session.current_date}`;
    if (state.session.phase === "morning") return `Morning execution — ${state.session.current_date}`;
    return `Replay completed — ${state.session.current_date}`;
  }, [state]);

  const createReplay = async (body: unknown) => {
    const created = await execute(() => api.createReplay(body));
    if (created) {
      setState(created);
      setShowCreate(false);
      setSelectedTicker(created.session.tickers[0] ?? "");
      await refreshLists();
    }
  };

  if ((!state && replays.length === 0 && definitions.length > 0) || showCreate) {
    return (
      <>
        {error && <div className="error-banner">{error}</div>}
        <CreateReplay onCreate={createReplay} onCancel={state ? () => setShowCreate(false) : undefined} />
      </>
    );
  }

  if (!state) return <div className="loading">Loading replay…</div>;

  return (
    <div className={`app phase-${state.session.phase}`}>
      <header className="app-header">
        <div>
          <span className="eyebrow">{state.session.name}</span>
          <h1>{phaseTitle}</h1>
        </div>
        <div className="header-stats">
          <span>Cash <strong>{state.session.cash.toLocaleString("sv-SE", { maximumFractionDigits: 0 })} SEK</strong></span>
          <span>Courtage <strong>{state.session.courtage_profile.name}</strong></span>
          <span>Barrier <strong>{state.session.barrier_policy.replaceAll("_", " ")}</strong></span>
          <select value={state.session.id} onChange={(event) => void loadReplay(event.target.value)}>
            {replays.map((replay) => <option value={replay.id} key={replay.id}>{replay.name}</option>)}
          </select>
          <button onClick={() => setShowCreate(true)}>New replay</button>
        </div>
      </header>
      {error && <div className="error-banner">{error}</div>}
      {busy && <div className="busy-bar" />}

      <main className="workspace">
        <aside className="left-column">
          <section className="panel ticker-list">
            <h2>Universe</h2>
            <input placeholder="Filter tickers" onChange={(event) => {
              const query = event.target.value.toUpperCase();
              const first = state.session.tickers.find((ticker) => ticker.includes(query));
              if (first && query) setSelectedTicker(first);
            }} />
            <div className="ticker-scroll">
              {state.session.tickers.map((ticker) => (
                <button key={ticker} className={ticker === selectedTicker ? "selected" : ""} onClick={() => setSelectedTicker(ticker)}>{ticker}</button>
              ))}
            </div>
          </section>
          <section className="panel watchlist-panel">
            <h2>Watchlist</h2>
            {state.watchlist.length === 0 && <p className="muted">No watched stocks.</p>}
            {state.watchlist.map((item) => (
              <div key={item.id} className="watchlist-row">
                <button onClick={() => setSelectedTicker(item.ticker)}>{item.ticker}</button>
                <span>{item.added_date}</span>
                <button className="danger-link" onClick={async () => acceptState(await execute(() => api.removeWatchlist(state.session.id, item.id)))}>×</button>
              </div>
            ))}
            <button onClick={async () => acceptState(await execute(() => api.addWatchlist(state.session.id, selectedTicker)))}>Watch {selectedTicker}</button>
          </section>
          <IndicatorPanel definitions={definitions} instances={instances} onChange={setInstances} />
        </aside>

        <section className="center-column">
          <MarketChart data={chart} definitions={definitions} instances={instances} />
          <PositionReview
            state={state}
            selectedTicker={selectedTicker}
            onSaveDecision={async (body) => acceptState(await execute(() => api.saveDecision(state.session.id, body)))}
            onFinalizeEvening={async () => acceptState(await execute(() => api.finalizeEvening(state.session.id)))}
            onReviseMorning={async (decisionId, body) => acceptState(await execute(() => api.reviseDecision(state.session.id, decisionId, body)))}
            onCompleteMorning={async () => acceptState(await execute(() => api.completeMorning(state.session.id)))}
          />
          <Metrics metrics={state.metrics} history={state.metric_history} scatter={state.position_scatter} />
          <AuditTrail fills={state.fills} events={state.events} />
        </section>

        <aside className="right-column">
          <Screener
            definitions={definitions}
            presets={presets}
            onRun={async (configuration: ScreeningConfiguration) => {
              const result = await execute(() => api.runScreen(state.session.id, configuration));
              return result?.results ?? [];
            }}
            onSavePreset={async (name, configuration) => {
              await execute(() => api.createPreset(name, configuration));
              setPresets(await api.presets());
            }}
            onDeletePreset={async (id) => {
              await execute(() => api.deletePreset(id));
              setPresets(await api.presets());
            }}
            onSelectTicker={setSelectedTicker}
            onAddWatchlist={async (ticker) => acceptState(await execute(() => api.addWatchlist(state.session.id, ticker)))}
          />
        </aside>
      </main>
    </div>
  );
}
