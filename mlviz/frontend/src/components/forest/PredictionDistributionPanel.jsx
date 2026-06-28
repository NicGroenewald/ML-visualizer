import Panel from "./Panel";
import { getTheme, FONT_MONO } from "../../theme";

function Metric({ label, value, T, accent = false }) {
  return (
    <div>
      <div style={{
        fontSize: accent ? 17 : 13,
        fontWeight: accent ? 600 : 500,
        color: accent ? T.accent : T.text,
        fontFamily: FONT_MONO,
      }}>
        {value}
      </div>
      <div style={{ fontSize: 10, color: T.textTertiary, marginTop: 2 }}>
        {label}
      </div>
    </div>
  );
}

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(3);
}

export default function PredictionDistributionPanel({ predictionDistribution, dark }) {
  const T = getTheme(dark);

  if (predictionDistribution === null) {
    return (
      <Panel title="prediction spread" dark={dark}>
        <div style={{ fontSize: 11, color: T.textTertiary, lineHeight: 1.5 }}>
          No query sample supplied — prediction spread is unavailable.
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="prediction spread" dark={dark}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>
        <Metric label="mean" value={fmt(predictionDistribution.mean)} T={T} accent />
        <Metric label="std" value={fmt(predictionDistribution.std)} T={T} />
        <Metric label="min" value={fmt(predictionDistribution.min)} T={T} />
        <Metric label="max" value={fmt(predictionDistribution.max)} T={T} />
      </div>
      <div style={{
        marginTop: 12,
        paddingTop: 10,
        borderTop: `1px solid ${T.separator}`,
        fontSize: 11,
        color: T.textTertiary,
        fontFamily: FONT_MONO,
      }}>
        {predictionDistribution.total_trees} tree predictions
      </div>
    </Panel>
  );
}
