import { useState, useEffect } from "react";
import MLVizTree from "./MLVizTree";

function initDark() {
  try {
    const stored = localStorage.getItem("mlviz-dark");
    if (stored !== null) return stored === "true";
  } catch {}
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
}

export default function App() {
  const [status, setStatus] = useState("loading");
  const [data,   setData]   = useState(null);
  const [dark,   setDark]   = useState(initDark);

  const toggleDark = () =>
    setDark((d) => {
      const next = !d;
      try { localStorage.setItem("mlviz-dark", String(next)); } catch {}
      return next;
    });

  useEffect(() => {
    fetch("/api/tree")
      .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
      .then((json) => { setData(json); setStatus("ready"); })
      .catch(() => setStatus("error"));
  }, []);

  const bg = dark ? "#000000" : "#F2F2F7";
  const centreStyle = {
    display: "flex", alignItems: "center", justifyContent: "center",
    height: "100vh", background: bg,
    fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
    fontSize: 14,
  };

  if (status === "loading")
    return <div style={{ ...centreStyle, color: dark ? "#AEAEB2" : "#8E8E93" }}>Loading…</div>;

  if (status === "error")
    return (
      <div style={{ ...centreStyle, color: dark ? "#FF453A" : "#FF3B30" }}>
        Could not load /api/tree — is the mlviz server running?
      </div>
    );

  return <MLVizTree data={data} dark={dark} toggleDark={toggleDark} />;
}
