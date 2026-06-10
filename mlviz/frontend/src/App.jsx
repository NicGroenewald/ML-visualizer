import { useState, useEffect } from "react";
import DecisionTreeView from "./views/DecisionTreeView";
import RandomForestView from "./views/RandomForestView";
import { getTheme, FONT_UI } from "./theme";

function initDark() {
  try {
    const stored = localStorage.getItem("mlviz-dark");
    if (stored !== null) return stored === "true";
  } catch {}
  return true; // dark by default — it's the tool's native theme
}

export default function App() {
  const [status, setStatus] = useState("loading");
  const [data, setData] = useState(null);
  const [dark, setDark] = useState(initDark);

  const toggleDark = () =>
    setDark((d) => {
      const next = !d;
      try { localStorage.setItem("mlviz-dark", String(next)); } catch {}
      return next;
    });

  useEffect(() => {
    fetch("/api/model")
      .then((r) => {
        if (r.ok) return r.json();
        // Legacy fallback: older servers only expose /api/tree
        return { model_type: "decision_tree", endpoint: "/api/tree" };
      })
      .catch(() => ({ model_type: "decision_tree", endpoint: "/api/tree" }))
      .then((manifest) =>
        fetch(manifest.endpoint).then((r) => {
          if (!r.ok) throw new Error(`HTTP ${r.status}`);
          return r.json();
        })
      )
      .then((json) => { setData(json); setStatus("ready"); })
      .catch(() => setStatus("error"));
  }, []);

  useEffect(() => {
    if (!data) return;

    const modelLabel =
      data.model_type === "decision_tree"
        ? "decision tree"
        : data.model_type === "random_forest"
          ? "random forest"
          : String(data.model_type ?? "model");

    if (data.dataset_name) {
      document.title = `mlviz · ${data.dataset_name} · ${modelLabel}`;
      return;
    }

    document.title = `mlviz · ${modelLabel}`;
  }, [data]);

  const T = getTheme(dark);
  const centreStyle = {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: "100vh", background: T.bg,
    fontFamily: FONT_UI,
    fontSize: 13,
  };

  if (status === "loading")
    return (
      <div style={centreStyle}>
        <span
          className="mlviz-loading-dot"
          style={{
            display: "inline-block",
            width: 8,
            height: 8,
            borderRadius: "50%",
            background: T.textSecondary,
            animation: "mlviz-pulse 900ms ease-in-out infinite",
          }}
        />
      </div>
    );

  if (status === "error")
    return (
      <div style={{ ...centreStyle, color: T.red }}>
        Could not reach the mlviz server — is it still running?
      </div>
    );

  if (data.model_type === "decision_tree")
    return <DecisionTreeView data={data} dark={dark} toggleDark={toggleDark} />;

  if (data.model_type === "random_forest")
    return <RandomForestView data={data} dark={dark} toggleDark={toggleDark} />;

  return (
    <div style={{ ...centreStyle, color: T.red }}>
      Unsupported model type: {String(data?.model_type)}
    </div>
  );
}
