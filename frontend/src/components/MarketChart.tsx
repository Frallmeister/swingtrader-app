import { useEffect, useRef } from "react";
import {
  CandlestickSeries,
  ColorType,
  createChart,
  createSeriesMarkers,
  HistogramSeries,
  LineSeries,
  type Time,
} from "lightweight-charts";
import type { ChartResponse, IndicatorDefinition, IndicatorInstance } from "../types";

interface Props {
  data: ChartResponse | null;
  definitions: IndicatorDefinition[];
  instances: IndicatorInstance[];
}

export function MarketChart({ data, definitions, instances }: Props) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current || !data) return;
    const chart = createChart(container.current, {
      autoSize: true,
      height: 620,
      layout: {
        background: { type: ColorType.Solid, color: "#10151d" },
        textColor: "#c9d2df",
        panes: { enableResize: true },
      },
      grid: {
        vertLines: { color: "#202a37" },
        horzLines: { color: "#202a37" },
      },
      timeScale: { borderColor: "#334155" },
      rightPriceScale: { borderColor: "#334155" },
    });
    const candles = chart.addSeries(CandlestickSeries, {
      upColor: "#2fbf83",
      downColor: "#ef5a67",
      wickUpColor: "#2fbf83",
      wickDownColor: "#ef5a67",
      borderVisible: false,
    });
    candles.setData(data.bars.map((bar) => ({ ...bar, time: bar.time as Time })));

    let nextPane = 1;
    const markers: Array<{ time: Time; position: "aboveBar" | "belowBar"; shape: "arrowDown" | "arrowUp"; text: string }> = [];
    data.indicator_groups.forEach((group, groupIndex) => {
      const instance = instances[groupIndex];
      const definition = definitions.find((item) => item.id === group.indicator_id);
      if (!instance || !definition) return;
      const separatePane = nextPane;
      let usedSeparatePane = false;
      definition.outputs.forEach((output) => {
        if (instance.hiddenOutputs.includes(output.id)) return;
        const values = group.outputs[output.id];
        if (!values) return;
        if (output.chart_style === "marker") {
          values.forEach((point) => {
            if (!point.value) return;
            const highMarker = output.id.includes("high");
            markers.push({
              time: point.time as Time,
              position: highMarker ? "aboveBar" : "belowBar",
              shape: highMarker ? "arrowDown" : "arrowUp",
              text: output.label,
            });
          });
          return;
        }
        const paneIndex = output.pane === "price" ? 0 : separatePane;
        usedSeparatePane ||= output.pane === "separate";
        const series = chart.addSeries(
          output.chart_style === "line" ? LineSeries : HistogramSeries,
          { title: output.label },
          paneIndex,
        );
        series.setData(values.map((point) => ({
          time: point.time as Time,
          value: typeof point.value === "boolean" ? (point.value ? 1 : 0) : point.value,
        })));
      });
      if (usedSeparatePane) nextPane += 1;
    });
    if (markers.length > 0) createSeriesMarkers(candles, markers);
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [data, definitions, instances]);

  if (!data) return <div className="chart-placeholder">Select a ticker to load its chart.</div>;
  return (
    <section className="chart-card">
      <div className="chart-title">
        <strong>{data.ticker}</strong>
        <span>{data.current_date}</span>
        {data.current_open != null && <span>Morning open: {data.current_open.toFixed(2)}</span>}
      </div>
      <div ref={container} className="chart-container" />
      <small className="attribution">
        Charts powered by <a href="https://www.tradingview.com/">TradingView Lightweight Charts™</a>
      </small>
    </section>
  );
}
