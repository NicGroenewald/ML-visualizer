import { useState } from "react";
import AppHeader, { Pill, ClassLegend, ThemeToggle } from "../components/AppHeader";
import Panel from "../components/forest/Panel";
import {
  KnnRegressionExplanation,
  KnnWeightedAveragePanel,
  KnnProjectionPlot,
  KnnRegressionTargetStrip,
  KnnVoteBarChart,
} from "../components/nonTree/NonTreeVisuals";
import { getTheme, getClassColors, FONT_MONO, FONT_UI } from "../theme";

const SIDEBAR_WIDTH = 320;

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

function MetricGrid({ items, T }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      {items.map((item) => (
        <div key={item.label}>
          <div style={{ fontSize: 14, fontWeight: 600, color: T.text, fontFamily: FONT_MONO }}>
            {fmt(item.value)}
          </div>
          <div style={{ fontSize: 10, color: T.textTertiary, marginTop: 3 }}>
            {item.label}
          </div>
        </div>
      ))}
    </div>
  );
}

function EmptyState({ children, T }) {
  return (
    <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>
      {children}
    </div>
  );
}

function NeighborTable({ neighbors, T, taskType }) {
  if (neighbors === null) {
    return <EmptyState T={T}>No query sample supplied - nearest-neighbor explanation is unavailable.</EmptyState>;
  }
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", minWidth: 620 }}>
        <thead>
          <tr>
            {["rank", "training row", "distance", "weight", taskType === "regression" ? "target" : "label"].map((label) => (
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
          {neighbors.map((n) => (
            <tr key={`${n.rank}-${n.training_index}`} style={{ borderTop: `1px solid ${T.separator}` }}>
              <td style={cellStyle(T)}>{n.rank}</td>
              <td style={cellStyle(T)}>{n.training_index}</td>
              <td style={cellStyle(T)}>{fmt(n.distance)}</td>
              <td style={cellStyle(T)}>{fmt(n.effective_weight)}</td>
              <td style={cellStyle(T)}>
                {taskType === "regression" ? fmt(n.target_value) : String(n.class_label)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function VotePanel({ data, T, dark }) {
  if (data.task_type === "regression") {
    return (
      <Panel title="prediction" dark={dark}>
        {data.prediction === null ? (
          <EmptyState T={T}>No query sample supplied - prediction is unavailable.</EmptyState>
        ) : (
          <MetricGrid T={T} items={[{ label: "predicted value", value: data.prediction.value }]} />
        )}
      </Panel>
    );
  }
  if (data.vote_distribution === null) {
    return (
      <Panel title="neighbor votes" dark={dark}>
        <EmptyState T={T}>No query sample supplied - neighbor votes are unavailable.</EmptyState>
      </Panel>
    );
  }
  return (
    <Panel title="neighbor votes" dark={dark}>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {data.vote_distribution.class_votes.map((item) => (
          <div key={item.class_index}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 3 }}>
              <span style={{ fontSize: 11, color: T.textSecondary }}>{String(item.class_label)}</span>
              <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO }}>
                {fmt(item.weight)} - {(item.fraction * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ height: 4, borderRadius: 2, background: T.surfaceSecondary, overflow: "hidden" }}>
              <div style={{
                height: "100%",
                width: `${item.fraction * 100}%`,
                borderRadius: 2,
                background: T.accent,
                transition: "width 220ms cubic-bezier(0.23, 1, 0.32, 1)",
              }} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
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

function firstRegressionNeighbor(data) {
  return data.neighbor_projection?.points?.find((point) => point.point_type === "neighbor")?.training_index ?? null;
}

export default function KnnView({ data, dark, toggleDark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const taskType = data.task_type ?? "classification";
  const isRegression = taskType === "regression";
  const summary = data.summary;
  const [tableOpen, setTableOpen] = useState(false);
  const [activeTrainingIndex, setActiveTrainingIndex] = useState(() => firstRegressionNeighbor(data));

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
            <Pill T={T} accent>k-nearest neighbors</Pill>
            <Pill T={T}>{taskType}</Pill>
            {data.dataset_name ? <Pill T={T}>{data.dataset_name}</Pill> : null}
            <Pill T={T}>k: {summary.n_neighbors}</Pill>
            <Pill T={T}>{summary.metric}</Pill>
            <Pill T={T}>weights: {summary.weights}</Pill>
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
          {isRegression ? (
            <KnnRegressionExplanation
              data={data}
              dark={dark}
              activeTrainingIndex={activeTrainingIndex}
              setActiveTrainingIndex={setActiveTrainingIndex}
            />
          ) : (
            <KnnProjectionPlot data={data} dark={dark} />
          )}
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
          <VotePanel data={data} T={T} dark={dark} />
          {isRegression && data.neighbor_projection ? (
            <Panel title="weighted average" dark={dark}>
              <KnnWeightedAveragePanel
                data={data}
                dark={dark}
                activeTrainingIndex={activeTrainingIndex}
                setActiveTrainingIndex={setActiveTrainingIndex}
                showHeading={false}
              />
            </Panel>
          ) : null}
          <Panel title="model summary" dark={dark}>
            <MetricGrid
              T={T}
              items={[
                { label: "neighbors", value: summary.n_neighbors },
                { label: "features", value: summary.n_features },
                { label: "algorithm", value: summary.algorithm },
                { label: "p", value: summary.p },
              ]}
            />
          </Panel>
          {isRegression ? (
            <KnnRegressionTargetStrip data={data} dark={dark} />
          ) : (
            <KnnVoteBarChart data={data} dark={dark} />
          )}
        </aside>
      </div>

      {/* Collapsible neighbor table */}
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
          <span>neighbors</span>
          <span style={{ fontSize: 9, marginTop: 1 }}>{tableOpen ? "▲" : "▼"}</span>
        </button>
        {tableOpen && (
          <div style={{
            maxHeight: 240,
            overflowY: "auto",
            padding: "0 16px 12px",
          }}>
            <NeighborTable neighbors={data.neighbors} T={T} taskType={taskType} />
          </div>
        )}
      </div>
    </div>
  );
}
