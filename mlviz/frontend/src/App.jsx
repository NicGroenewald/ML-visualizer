import { useState, useEffect } from "react";
import MLVizTree from "./MLVizTree";

const centre = {
  display: "flex", alignItems: "center", justifyContent: "center",
  height: "100vh",
  fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
  fontSize: 14,
};

export default function App() {
  const [status, setStatus] = useState("loading");
  const [data,   setData]   = useState(null);

  useEffect(() => {
    fetch("/api/tree")
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((json) => { setData(json); setStatus("ready"); })
      .catch(()    => setStatus("error"));
  }, []);

  if (status === "loading") return (
    <div style={{ ...centre, color: "#8E8E93" }}>Loading…</div>
  );

  if (status === "error") return (
    <div style={{ ...centre, color: "#FF3B30" }}>
      Could not load /api/tree — is the mlviz server running?
    </div>
  );

  return <MLVizTree data={data} />;
}
