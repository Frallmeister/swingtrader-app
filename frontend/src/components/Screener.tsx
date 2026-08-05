import { useMemo, useState } from "react";
import type {
  IndicatorDefinition,
  Operand,
  ScreeningConfiguration,
  ScreeningExpression,
  ScreeningPreset,
  ScreeningRule,
} from "../types";
import { IndicatorParameterFields } from "./IndicatorParameterFields";

const columns = ["open", "high", "low", "close", "adjusted_close", "volume"];

function defaultOperand(): Operand {
  return { kind: "column", column: "close" };
}

function defaultExpression(): ScreeningExpression {
  return { left: defaultOperand(), operation: "identity", lookback_sessions: 1, aggregation: "latest" };
}

interface OperandEditorProps {
  value: Operand;
  definitions: IndicatorDefinition[];
  onChange: (value: Operand) => void;
}

function OperandEditor({ value, definitions, onChange }: OperandEditorProps) {
  const definition = definitions.find((item) => item.id === value.indicator_id);
  const chooseIndicator = (indicatorId: string) => {
    const selected = definitions.find((item) => item.id === indicatorId);
    if (!selected) return;
    onChange({
      kind: "indicator",
      indicator_id: selected.id,
      source: selected.default_source,
      output: selected.outputs[0]?.id,
      parameters: Object.fromEntries(selected.parameters.map((parameter) => [parameter.name, parameter.default])),
    });
  };

  return (
    <div className="operand-editor">
      <select
        value={value.kind}
        onChange={(event) => event.target.value === "column"
          ? onChange(defaultOperand())
          : chooseIndicator(definitions[0]?.id ?? "")}
      >
        <option value="column">Market column</option>
        <option value="indicator">Indicator output</option>
      </select>
      {value.kind === "column" ? (
        <select value={value.column} onChange={(event) => onChange({ kind: "column", column: event.target.value })}>
          {columns.map((column) => <option key={column}>{column}</option>)}
        </select>
      ) : (
        <>
          <select value={value.indicator_id} onChange={(event) => chooseIndicator(event.target.value)}>
            {definitions.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}
          </select>
          {definition?.input_kind === "series" && (
            <select value={value.source ?? "close"} onChange={(event) => onChange({ ...value, source: event.target.value })}>
              {columns.map((column) => <option key={column}>{column}</option>)}
            </select>
          )}
          {definition && (
            <IndicatorParameterFields
              parameters={definition.parameters}
              values={value.parameters ?? {}}
              onChange={(parameters) => onChange({ ...value, parameters })}
            />
          )}
          <select value={value.output} onChange={(event) => onChange({ ...value, output: event.target.value })}>
            {definition?.outputs.map((output) => <option value={output.id} key={output.id}>{output.label}</option>)}
          </select>
        </>
      )}
    </div>
  );
}

interface Props {
  definitions: IndicatorDefinition[];
  presets: ScreeningPreset[];
  onRun: (configuration: ScreeningConfiguration) => Promise<Array<{ ticker: string; values: Record<string, number> }>>;
  onSavePreset: (name: string, configuration: ScreeningConfiguration) => Promise<void>;
  onDeletePreset: (id: string) => Promise<void>;
  onSelectTicker: (ticker: string) => void;
  onAddWatchlist: (ticker: string) => Promise<void>;
}

export function Screener({ definitions, presets, onRun, onSavePreset, onDeletePreset, onSelectTicker, onAddWatchlist }: Props) {
  const [rules, setRules] = useState<ScreeningRule[]>([]);
  const [expression, setExpression] = useState<ScreeningExpression>(defaultExpression());
  const [comparison, setComparison] = useState<ScreeningRule["comparison"]>("gte");
  const [threshold, setThreshold] = useState(1);
  const [maximum, setMaximum] = useState(1.1);
  const [results, setResults] = useState<Array<{ ticker: string; values: Record<string, number> }>>([]);
  const [presetName, setPresetName] = useState("");
  const [sortEnabled, setSortEnabled] = useState(true);

  const configuration = useMemo<ScreeningConfiguration>(() => ({
    rules,
    sort: sortEnabled && rules.length > 0 ? [{ expression: rules[0].expression, direction: "desc" }] : [],
    exclude_owned: true,
    exclude_pending_buys: true,
  }), [rules, sortEnabled]);

  const loadPreset = (id: string) => {
    const preset = presets.find((item) => item.id === id);
    if (!preset) return;
    setRules(preset.configuration.rules);
    setPresetName(preset.name);
    setSortEnabled(preset.configuration.sort.length > 0);
  };

  const addRule = () => {
    const rule: ScreeningRule = comparison === "between"
      ? { expression, comparison, minimum: threshold, maximum }
      : { expression, comparison, value: threshold };
    setRules([...rules, rule]);
  };

  return (
    <section className="panel screener-panel">
      <div className="panel-heading">
        <h2>Indicator screener</h2>
        <select defaultValue="" onChange={(event) => loadPreset(event.target.value)}>
          <option value="">Load saved screen…</option>
          {presets.map((preset) => <option key={preset.id} value={preset.id}>{preset.name}</option>)}
        </select>
      </div>
      <div className="screen-rule-builder">
        <OperandEditor value={expression.left} definitions={definitions} onChange={(left) => setExpression({ ...expression, left })} />
        <select value={expression.operation} onChange={(event) => {
          const operation = event.target.value as ScreeningExpression["operation"];
          setExpression({ ...expression, operation, right: operation === "identity" ? undefined : expression.right ?? defaultOperand() });
        }}>
          <option value="identity">value</option><option value="divide">divided by</option>
          <option value="subtract">minus</option><option value="add">plus</option><option value="multiply">multiplied by</option>
        </select>
        {expression.operation !== "identity" && expression.right && (
          <OperandEditor value={expression.right} definitions={definitions} onChange={(right) => setExpression({ ...expression, right })} />
        )}
        <select value={comparison} onChange={(event) => setComparison(event.target.value as ScreeningRule["comparison"])}>
          <option value="gte">≥</option><option value="gt">&gt;</option><option value="lte">≤</option>
          <option value="lt">&lt;</option><option value="eq">=</option><option value="between">Between</option>
        </select>
        <input aria-label="Threshold" type="number" step="any" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
        {comparison === "between" && <input aria-label="Maximum" type="number" step="any" value={maximum} onChange={(event) => setMaximum(Number(event.target.value))} />}
        <label>Lookback<input type="number" min={1} max={252} value={expression.lookback_sessions} onChange={(event) => setExpression({ ...expression, lookback_sessions: Number(event.target.value) })} /></label>
        <select value={expression.aggregation} onChange={(event) => setExpression({ ...expression, aggregation: event.target.value as ScreeningExpression["aggregation"] })}>
          <option value="latest">Latest</option><option value="maximum">Maximum</option><option value="minimum">Minimum</option><option value="mean">Mean</option>
        </select>
        <button onClick={addRule}>Add rule</button>
      </div>
      <ol className="rule-list">
        {rules.map((rule, index) => (
          <li key={index}>
            Rule {index + 1}: {rule.comparison} {rule.comparison === "between" ? `${rule.minimum}–${rule.maximum}` : String(rule.value)}
            <button className="danger-link" onClick={() => setRules(rules.filter((_, itemIndex) => itemIndex !== index))}>Remove</button>
          </li>
        ))}
      </ol>
      <label className="checkbox-row"><input type="checkbox" checked={sortEnabled} onChange={(event) => setSortEnabled(event.target.checked)} />Sort by the first rule expression</label>
      <div className="toolbar">
        <button className="primary" onClick={async () => setResults(await onRun(configuration))}>Run screen</button>
        <input placeholder="Preset name" value={presetName} onChange={(event) => setPresetName(event.target.value)} />
        <button disabled={!presetName} onClick={() => onSavePreset(presetName, configuration)}>Save as preset</button>
      </div>
      {presets.length > 0 && <details><summary>Manage saved screens</summary>{presets.map((preset) => (
        <div key={preset.id} className="saved-row"><span>{preset.name}</span><button className="danger-link" onClick={() => onDeletePreset(preset.id)}>Delete</button></div>
      ))}</details>}
      <div className="screen-results">
        {results.map((result) => (
          <div key={result.ticker} className="screen-result" onClick={() => onSelectTicker(result.ticker)}>
            <strong>{result.ticker}</strong>
            <span>{Object.values(result.values).filter((value) => value != null).map((value) => Number(value).toFixed(3)).join(" · ")}</span>
            <button onClick={(event) => { event.stopPropagation(); void onAddWatchlist(result.ticker); }}>Watch</button>
          </div>
        ))}
      </div>
    </section>
  );
}
