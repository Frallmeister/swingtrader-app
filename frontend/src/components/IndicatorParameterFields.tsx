import type { IndicatorParameter } from "../types";

export function parseParameterValue(parameter: IndicatorParameter, raw: string | boolean): unknown {
  if (parameter.kind === "boolean") return Boolean(raw);
  if (parameter.kind === "integer") return Number.parseInt(String(raw), 10);
  if (parameter.kind === "number") return Number.parseFloat(String(raw));
  if (parameter.kind === "integer_tuple") {
    return String(raw).split(",").map((value) => Number.parseInt(value.trim(), 10));
  }
  return raw;
}

function displayValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(", ");
  return value == null ? "" : String(value);
}

interface Props {
  parameters: IndicatorParameter[];
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
}

export function IndicatorParameterFields({ parameters, values, onChange }: Props) {
  return (
    <div className="parameter-grid">
      {parameters.map((parameter) => {
        const current = values[parameter.name];
        if (parameter.kind === "boolean") {
          return (
            <label key={parameter.name} className="checkbox-row">
              <input
                type="checkbox"
                checked={Boolean(current)}
                onChange={(event) => onChange({ ...values, [parameter.name]: event.target.checked })}
              />
              {parameter.label}
            </label>
          );
        }
        if (parameter.kind === "choice") {
          return (
            <label key={parameter.name}>
              {parameter.label}
              <select
                value={displayValue(current)}
                onChange={(event) => onChange({
                  ...values,
                  [parameter.name]: parseParameterValue(parameter, event.target.value),
                })}
              >
                {parameter.choices.map((choice) => (
                  <option key={String(choice)} value={String(choice)}>{String(choice)}</option>
                ))}
              </select>
            </label>
          );
        }
        return (
          <label key={parameter.name}>
            {parameter.label}
            <input
              type={parameter.kind === "integer_tuple" || parameter.kind === "text" ? "text" : "number"}
              value={displayValue(current)}
              step={parameter.kind === "integer" ? 1 : "any"}
              onChange={(event) => onChange({
                ...values,
                [parameter.name]: parseParameterValue(parameter, event.target.value),
              })}
            />
          </label>
        );
      })}
    </div>
  );
}
