import type { IndicatorDefinition, IndicatorInstance } from "../types";
import { IndicatorParameterFields } from "./IndicatorParameterFields";

interface Props {
  definitions: IndicatorDefinition[];
  instances: IndicatorInstance[];
  onChange: (instances: IndicatorInstance[]) => void;
}

export function IndicatorPanel({ definitions, instances, onChange }: Props) {
  const add = (indicatorId: string) => {
    const definition = definitions.find((item) => item.id === indicatorId);
    if (!definition) return;
    onChange([
      ...instances,
      {
        instanceId: crypto.randomUUID(),
        indicator_id: definition.id,
        source: definition.default_source,
        parameters: Object.fromEntries(
          definition.parameters.map((parameter) => [parameter.name, parameter.default]),
        ),
        hiddenOutputs: [],
      },
    ]);
  };

  const update = (instanceId: string, changes: Partial<IndicatorInstance>) => {
    onChange(
      instances.map((instance) =>
        instance.instanceId === instanceId ? { ...instance, ...changes } : instance,
      ),
    );
  };

  return (
    <section className="panel indicator-panel">
      <div className="panel-heading">
        <h2>Indicators</h2>
        <select defaultValue="" onChange={(event) => add(event.target.value)}>
          <option value="" disabled>
            Add indicator…
          </option>
          {definitions.map((definition) => (
            <option key={definition.id} value={definition.id}>
              {definition.label}
            </option>
          ))}
        </select>
      </div>

      {instances.length === 0 && <p className="muted">Add an indicator to calibrate and plot it.</p>}
      {instances.map((instance) => {
        const definition = definitions.find((item) => item.id === instance.indicator_id);
        if (!definition) return null;
        return (
          <details key={instance.instanceId} open className="indicator-instance">
            <summary>{definition.label}</summary>
            {definition.input_kind === "series" && (
              <label>
                Input column
                <select
                  value={instance.source ?? "close"}
                  onChange={(event) => update(instance.instanceId, { source: event.target.value })}
                >
                  {['open', 'high', 'low', 'close', 'adjusted_close', 'volume'].map((column) => (
                    <option key={column}>{column}</option>
                  ))}
                </select>
              </label>
            )}
            <IndicatorParameterFields
              parameters={definition.parameters}
              values={instance.parameters}
              onChange={(parameters) => update(instance.instanceId, { parameters })}
            />
            <fieldset>
              <legend>Visible outputs</legend>
              {definition.outputs.map((output) => (
                <label key={output.id} className="checkbox-row">
                  <input
                    type="checkbox"
                    checked={!instance.hiddenOutputs.includes(output.id)}
                    onChange={(event) =>
                      update(instance.instanceId, {
                        hiddenOutputs: event.target.checked
                          ? instance.hiddenOutputs.filter((id) => id !== output.id)
                          : [...instance.hiddenOutputs, output.id],
                      })
                    }
                  />
                  {output.label}
                </label>
              ))}
            </fieldset>
            <button
              className="danger-link"
              onClick={() => onChange(instances.filter((item) => item.instanceId !== instance.instanceId))}
            >
              Remove
            </button>
          </details>
        );
      })}
    </section>
  );
}
