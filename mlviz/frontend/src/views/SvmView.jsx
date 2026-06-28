import { useState } from "react";
import AppHeader, { Pill, ClassLegend, ThemeToggle } from "../components/AppHeader";
import Panel from "../components/forest/Panel";
import {
  SvmDecisionScoreChart,
  SvmProjectionPlot,
  SvmRegressionValueStrip,
} from "../components/nonTree/NonTreeVisuals";
import { getTheme, getClassColors, FONT_MONO, FONT_UI } from "../theme";

const SIDEBAR_WIDTH = 320;

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

function EmptyState({ children, T }) {
  return <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>{children}</div>;
}

function MetricGrid({ items, T }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {items.map((item) => (
        <div key={item.label}>
          <div style={{ fontSize: 14, fontWeight: 600, color: T.text, fontFamily: FONT_MONO }}>
            {fmt(item.value)}
          </div>
          <div style={{ fontSize: 10, color: T.textTertiary, marginTop: 3 }}>{item.label}</div>
        </div>
      ))}
    </div>
  );
}

function cellStyle(T) {
  return {
    padding: "9px 10px 9px 0",
    color: T.textSecondary,
    fontSize: 12,
    fontFamily: FONT_MONO,
    whiteSpace: "nowrap",
  };
}

function SupportVectorTable({ supportVectors, T, taskType }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 680 }}>
        <thead>
          <tr>
            {["sv index", "training row", taskType === "regression" ? "target" : "label", "features"].map((label) => (
              <th key={label} style={{
                textAlign: "left",
                padding: "0 10px 8px 0",
                fontSize: 10,
                color: T.textTertiary,
                fontWeight: 500,
                textTransform: "uppercase",
                letterSpacing: "0.7px",
                whiteSpace: "nowrap",
              }}>
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {supportVectors.map((v) => (
            <tr key={`${v.support_vector_index}-${v.training_index}`} style={{ borderTop: `1px solid ${T.separator}` }}>
              <td style={cellStyle(T)}>{v.support_vector_index}</td>
              <td style={cellStyle(T)}>{v.training_index}</td>
              <td style={cellStyle(T)}>
                {taskType === "regression" ? fmt(v.target_value) : String(v.class_label)}
              </td>
              <td style={{ ...cellStyle(T), whiteSpace: "normal", lineHeight: 1.45 }}>
                {v.feature_values.map((f) => `${f.feature_name}: ${fmt(f.value)}`).join(" · ")}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function PredictionPanel({ data, T, dark }) {
  if (data.prediction === null) {
    return (
      <Panel title="query prediction" dark={dark}>
        <EmptyState T={T}>No query sample supplied - prediction details are unavailable.</EmptyState>
      </Panel>
    );
  }
  if (data.task_type === "regression") {
    return (
      <Panel title="query prediction" dark={dark}>
        <MetricGrid T={T} items={[{ label: "predicted value", value: data.prediction.value }]} />
      </Panel>
    );
  }
  return (
    <Panel title="query prediction" dark={dark}>
      <MetricGrid
        T={T}
        items={[
          { label: "class", value: data.prediction.class_label },
          { label: "class index", value: data.prediction.class_index },
        ]}
      />
      <div style={{
        marginTop: 12,
        paddingTop: 10,
        borderTop: `1px solid ${T.separator}`,
        color: T.textTertiary,
        fontSize: 11,
        fontFamily: FONT_MONO,
        lineHeight: 1.5,
      }}>
        decision {Array.isArray(data.prediction.decision_function)
          ? data.prediction.decision_function.map(fmt).join(", ")
          : fmt(data.prediction.decision_function)}
      </div>
    </Panel>
  );
}

export default function SvmView({ data, dark, toggleDark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const taskType = data.task_type ?? "classification";
  const isRegression = taskType === "regression";
  const summary = data.summary;
  const [tableOpen, setTableOpen] = useState(false);

  return (
    <div style={{
      background: T.bg,
      display: "flex",
      flexDirection: "column",
      height: "100vh",
      fontFamily: FONT_UI,
    }}>
      <AppHeader
        dark={dark}
        pills={
          <>
            <Pill T={T} accent>support vector machine</Pill>
            <Pill T={T}>{taskType}</Pill>
            {data.dataset_name ? <Pill T={T}>{data.dataset_name}</Pill> : null}
            <Pill T={T}>{summary.kernel}</Pill>
            <Pill T={T}>C: {fmt(summary.C)}</Pill>
            <Pill T={T}>support: {summary.support_vector_count}</Pill>
          </>
        }
        right={
          <>
            {!isRegression && (
              <ClassLegend classes={data.classes} classColors={classColors} T={T} />
            )}
            <ThemeToggle dark={dark} toggleDark={toggleDark} T={T} />
          </>
        }
      />

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <main style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          minHeight: 0,
        }}>
          <SvmProjectionPlot data={data} dark={dark} />
        </main>

        <aside style={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          borderLeft: `1px solid ${T.border}`,
          background: T.bg,
          overflowY: "auto",
          padding: 12,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}>
          <PredictionPanel data={data} T={T} dark={dark} />
          <Panel title="model summary" dark={dark}>
            <MetricGrid
              T={T}
              items={[
                { label: "features", value: summary.n_features },
                { label: "gamma", value: summary.gamma },
                { label: "degree", value: summary.degree },
                { label: isRegression ? "epsilon" : "coef0", value: isRegression ? summary.epsilon : summary.coef0 },
              ]}
            />
          </Panel>
          {isRegression ? (
            <SvmRegressionValueStrip data={data} dark={dark} />
          ) : (
            <SvmDecisionScoreChart data={data} dark={dark} />
          )}
        </aside>
      </div>

      {/* Collapsible support vector table */}
      <div style={{ borderTop: `1px solid ${T.border}`, background: T.surface }}>
        <button
          onClick={() => setTableOpen((v) => !v)}
          style={{
            width: "100%",
            padding: "8px 16px",
            background: "none",
            border: "none",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 6,
            color: T.textSecondary,
            fontSize: 11,
            fontWeight: 500,
            textTransform: "uppercase",
            letterSpacing: "0.7px",
            fontFamily: FONT_UI,
          }}
        >
          <span>support vectors</span>
          <span style={{ fontSize: 9, marginTop: 1 }}>{tableOpen ? "▲" : "▼"}</span>
        </button>
        {tableOpen && (
          <div style={{
            maxHeight: 240,
            overflowY: "auto",
            padding: "0 16px 12px",
          }}>
            <SupportVectorTable supportVectors={data.support_vectors} T={T} taskType={taskType} />
          </div>
        )}
      </div>
    </div>
  );
}
