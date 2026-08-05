export type ReplayPhase = "evening" | "morning" | "completed";
export type DecisionAction = "keep" | "sell" | "reduce" | "buy";

export interface ReplaySession {
  id: string;
  name: string;
  provider: string;
  tickers: string[];
  current_date: string;
  start_date: string;
  end_date: string;
  phase: ReplayPhase;
  status: string;
  initial_cash: number;
  cash: number;
  courtage_profile: { name: string };
  barrier_policy: string;
}

export interface Position {
  id: string;
  ticker: string;
  opened_date: string;
  entry_price: number;
  quantity: number;
  initial_quantity: number;
  stop_price: number | null;
  target_price: number | null;
  realized_pnl: number;
  realized_r: number;
  status: "open" | "closed";
}

export interface Decision {
  id: string;
  position_id: string | null;
  ticker: string;
  action: DecisionAction;
  quantity: number | null;
  allocation_sek: number | null;
  stop_price: number | null;
  target_price: number | null;
  status: string;
  note: string | null;
}

export interface MetricSnapshot {
  trading_date: string;
  total_return: number;
  expectancy_r: number | null;
  win_rate: number | null;
  cumulative_r: number;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  equity: number;
}

export interface ReplayState {
  session: ReplaySession;
  positions: Position[];
  pending_decisions: Decision[];
  watchlist: WatchlistItem[];
  metrics: MetricSnapshot | null;
  metric_history: MetricSnapshot[];
  outstanding_position_ids: string[];
  morning_open_prices: Record<string, number>;
  position_scatter: PositionScatterPoint[];
  fills: ReplayFill[];
  events: ReplayEvent[];
}


export interface ReplayFill {
  id: string;
  ticker: string;
  trading_date: string;
  side: "buy" | "sell";
  quantity: number;
  price: number;
  courtage: number;
  reason: string;
}

export interface ReplayEvent {
  id: string;
  sequence_number: number;
  event_type: string;
  trading_date: string;
  phase: ReplayPhase;
  ticker: string | null;
  payload: Record<string, unknown>;
}

export interface WatchlistItem {
  id: string;
  ticker: string;
  added_date: string;
  note: string | null;
}

export interface PositionScatterPoint {
  ticker: string;
  days_owned: number;
  simple_return: number;
  annualized_simple_return: number;
}

export interface IndicatorParameter {
  name: string;
  label: string;
  kind: "integer" | "number" | "boolean" | "choice" | "integer_tuple" | "text";
  default: unknown;
  required: boolean;
  choices: unknown[];
}

export interface IndicatorOutputDefinition {
  id: string;
  label: string;
  chart_style: "line" | "histogram" | "marker";
  pane: "price" | "separate";
}

export interface IndicatorDefinition {
  id: string;
  label: string;
  input_kind: "series" | "frame";
  default_source: string | null;
  parameters: IndicatorParameter[];
  outputs: IndicatorOutputDefinition[];
}

export interface IndicatorInstance {
  instanceId: string;
  indicator_id: string;
  parameters: Record<string, unknown>;
  source?: string | null;
  hiddenOutputs: string[];
}

export interface ChartDataPoint {
  time: string;
  value: number | boolean;
}

export interface ChartResponse {
  ticker: string;
  phase: ReplayPhase;
  current_date: string;
  current_open: number | null;
  bars: Array<{
    time: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number | null;
  }>;
  indicator_groups: Array<{
    indicator_id: string;
    parameters: Record<string, unknown>;
    source: string | null;
    outputs: Record<string, ChartDataPoint[]>;
  }>;
}

export interface Operand {
  kind: "column" | "indicator";
  column?: string;
  indicator_id?: string;
  parameters?: Record<string, unknown>;
  source?: string | null;
  output?: string;
}

export interface ScreeningExpression {
  left: Operand;
  operation: "identity" | "divide" | "subtract" | "add" | "multiply";
  right?: Operand;
  lookback_sessions: number;
  aggregation: "latest" | "maximum" | "minimum" | "mean";
}

export interface ScreeningRule {
  expression: ScreeningExpression;
  comparison: "gt" | "gte" | "lt" | "lte" | "between" | "eq";
  value?: number | boolean;
  minimum?: number;
  maximum?: number;
}

export interface ScreeningConfiguration {
  name?: string;
  preset_id?: string;
  rules: ScreeningRule[];
  sort: Array<{ expression: ScreeningExpression; direction: "asc" | "desc" }>;
  exclude_owned: boolean;
  exclude_pending_buys: boolean;
}

export interface ScreeningPreset {
  id: string;
  name: string;
  description: string | null;
  configuration: ScreeningConfiguration;
}
