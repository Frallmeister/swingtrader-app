import type {
  ChartResponse,
  IndicatorDefinition,
  ReplayState,
  ScreeningConfiguration,
  ScreeningPreset,
} from "../types";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...options?.headers },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(body.detail ?? response.statusText);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  indicators: () => request<IndicatorDefinition[]>("/api/indicators"),
  replays: () => request<Array<ReplayState["session"]>>("/api/replays"),
  replay: (id: string) => request<ReplayState>(`/api/replays/${id}`),
  createReplay: (body: unknown) =>
    request<ReplayState>("/api/replays", { method: "POST", body: JSON.stringify(body) }),
  saveDecision: (replayId: string, body: unknown) =>
    request<ReplayState>(`/api/replays/${replayId}/decisions`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  finalizeEvening: (replayId: string) =>
    request<ReplayState>(`/api/replays/${replayId}/finalize-evening`, { method: "POST" }),
  reviseDecision: (replayId: string, decisionId: string, body: unknown) =>
    request<ReplayState>(`/api/replays/${replayId}/decisions/${decisionId}`, {
      method: "PATCH",
      body: JSON.stringify(body),
    }),
  completeMorning: (replayId: string) =>
    request<ReplayState>(`/api/replays/${replayId}/complete-morning`, { method: "POST" }),
  chart: (replayId: string, ticker: string, body: unknown) =>
    request<ChartResponse>(`/api/replays/${replayId}/chart/${ticker}`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  runScreen: (replayId: string, configuration: ScreeningConfiguration) =>
    request<{ run_id: string; results: Array<{ ticker: string; values: Record<string, number> }> }>(
      `/api/replays/${replayId}/screen`,
      { method: "POST", body: JSON.stringify(configuration) },
    ),
  addWatchlist: (replayId: string, ticker: string, note?: string) =>
    request<ReplayState>(`/api/replays/${replayId}/watchlist`, {
      method: "POST",
      body: JSON.stringify({ ticker, note }),
    }),
  removeWatchlist: (replayId: string, itemId: string) =>
    request<ReplayState>(`/api/replays/${replayId}/watchlist/${itemId}`, { method: "DELETE" }),
  presets: () => request<ScreeningPreset[]>("/api/screening-presets"),
  createPreset: (name: string, configuration: ScreeningConfiguration) =>
    request<ScreeningPreset>("/api/screening-presets", {
      method: "POST",
      body: JSON.stringify({ name, configuration }),
    }),
  deletePreset: (id: string) =>
    request<void>(`/api/screening-presets/${id}`, { method: "DELETE" }),
};
