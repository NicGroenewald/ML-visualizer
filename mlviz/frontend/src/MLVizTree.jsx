import { useState, useEffect, useMemo, useRef } from "react";

// ─── Design tokens (Apple HIG light + dark) ───────────────────────────────────
const LIGHT = {
  bg:               "#F2F2F7",
  surface:          "#FFFFFF",
  surfaceSecondary: "#F2F2F7",
  border:           "#3C3C43",
  borderHover:      "#1C1C1E",
  text:             "#1C1C1E",
  textSecondary:    "#8E8E93",
  textTertiary:     "#AEAEB2",
  separator:        "#E5E5EA",
  blue:             "#007AFF",
  blueFill:         "#EBF5FF",
  green:            "#34C759",
  greenFill:        "#EEFBF1",
  orange:           "#FF9F0A",
  orangeFill:       "#FFF4E5",
  red:              "#FF3B30",
  classFill: ["#EBF5FF", "#EEFBF1", "#FFF4E5", "#F5EFFE", "#FFF0F3", "#E6FAFA"],
};

const DARK = {
  bg:               "#000000",
  surface:          "#1C1C1E",
  surfaceSecondary: "#2C2C2E",
  border:           "#48484A",
  borderHover:      "#AEAEB2",
  text:             "#FFFFFF",
  textSecondary:    "#AEAEB2",
  textTertiary:     "#636366",
  separator:        "#38383A",
  blue:             "#0A84FF",
  blueFill:         "#001E3C",
  green:            "#30D158",
  greenFill:        "#0A2A10",
  orange:           "#FF9F0A",
  orangeFill:       "#2A1500",
  red:              "#FF453A",
  classFill: ["#001E3C", "#0A2A10", "#2A1500", "#1A0030", "#30000E", "#002020"],
};

const CLASS_COLORS = ["#007AFF", "#34C759", "#FF9F0A", "#AF52DE", "#FF375F", "#00C7BE"];

// ─── Helpers ──────────────────────────────────────────────────────────────────
const sf = (f) => f ? f.replace(" (cm)", "") : "";
const clampLabel = (s, max = 19) =>
  s && s.length > max ? s.slice(0, max - 1) + "…" : (s ?? "");

// ─── Layout algorithm ─────────────────────────────────────────────────────────
const NW = 168, NH = 52, LH = 112, HG = 28, HP = 44, VP = 28;

function computeLayout(nodes) {
  const map = {};
  nodes.forEach((n) => { map[n.node_id] = { ...n }; });

  if (nodes.length === 0) return { map: {}, svgW: 0, svgH: 0 };

  let leafCount = 0;
  function setX(id) {
    const n = map[id];
    if (n.leaf) { n._x = leafCount++; return; }
    setX(n.left_child);
    setX(n.right_child);
    n._x = (map[n.left_child]._x + map[n.right_child]._x) / 2;
  }
  function setDepth(id, d) {
    map[id]._d = d;
    const n = map[id];
    if (!n.leaf) { setDepth(n.left_child, d + 1); setDepth(n.right_child, d + 1); }
  }

  setX(0);
  setDepth(0, 0);

  const spacing = NW + HG;
  Object.values(map).forEach((n) => {
    n.px = n._x * spacing + HP;
    n.py = n._d * LH + VP;
    n.cx = n.px + NW / 2;
    n.cy = n.py + NH / 2;
  });

  const allNodes = Object.values(map);
  const svgW = (allNodes.length ? Math.max(...allNodes.map((n) => n.px)) : 0) + NW + HP;
  const svgH = (allNodes.length ? Math.max(...allNodes.map((n) => n._d)) : 0) * LH + NH + VP * 2;

  return { map, svgW, svgH };
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function Pill({ children, T }) {
  return (
    <span style={{
      fontSize: 11, padding: "3px 10px", borderRadius: 20,
      background: T.surfaceSecondary, border: `0.5px solid ${T.border}`,
      color: T.textSecondary, letterSpacing: "0.1px",
    }}>
      {children}
    </span>
  );
}

function TRow({ label, value, mono = true, T }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2.5px 0" }}>
      <span style={{ fontSize: 11, color: T.textSecondary }}>{label}</span>
      <span style={{ fontSize: 11, color: T.text, fontFamily: mono ? '"SF Mono", Menlo, monospace' : "inherit" }}>
        {value}
      </span>
    </div>
  );
}

function Tooltip({ node, map, classes, T, classDot, classFill }) {
  if (!node) return null;
  const p = node.prediction;
  return (
    <div style={{
      background: T.surface,
      border: `0.5px solid ${T.separator}`,
      borderRadius: 13,
      padding: "13px 15px",
      minWidth: 182,
      boxSizing: "border-box",
    }}>
      <div style={{
        fontSize: 10, color: T.textTertiary, letterSpacing: "0.7px",
        textTransform: "uppercase", marginBottom: 5,
      }}>
        {node.leaf ? "leaf node" : "split node"}
      </div>

      {node.leaf ? (
        <>
          <div style={{
            fontSize: 13, fontWeight: 500, color: classDot[p] ?? T.textSecondary,
            marginBottom: 10, display: "flex", alignItems: "center", gap: 6,
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: classDot[p] ?? T.textSecondary,
              display: "inline-block", flexShrink: 0,
            }} />
            {classes[p]}
          </div>
          <TRow label="gini impurity" value={node.gini.toFixed(3)} T={T} />
          <TRow label="samples"       value={node.n_samples}        T={T} />
          <div style={{ height: "0.5px", background: T.separator, margin: "8px 0" }} />
          {classes.map((c, idx) => (
            <div key={c} style={{ display: "flex", justifyContent: "space-between", padding: "2.5px 0" }}>
              <span style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: T.textSecondary }}>
                <span style={{
                  width: 5, height: 5, borderRadius: "50%",
                  background: classDot[idx] ?? T.textSecondary,
                  display: "inline-block",
                }} />
                class {c}
              </span>
              <span style={{ fontSize: 11, fontFamily: '"SF Mono", Menlo, monospace', color: T.text }}>
                {node.counts?.[idx] ?? 0}
              </span>
            </div>
          ))}
        </>
      ) : (
        <>
          <div style={{
            fontSize: 13, fontWeight: 500, color: T.text, marginBottom: 10,
            fontFamily: '"SF Mono", Menlo, monospace',
          }}>
            {sf(node.feature)} ≤ {node.threshold}
          </div>
          <TRow label="gini impurity" value={node.gini.toFixed(3)} T={T} />
          <TRow label="samples"       value={node.n_samples}        T={T} />
          <div style={{ height: "0.5px", background: T.separator, margin: "8px 0" }} />
          <TRow label="left branch"  value={map[node.left_child]?.n_samples}  T={T} />
          <TRow label="right branch" value={map[node.right_child]?.n_samples} T={T} />
        </>
      )}
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function MLVizTree({ data, dark, toggleDark }) {
  const T           = dark ? DARK : LIGHT;
  const classDot    = CLASS_COLORS;
  const classFill   = T.classFill;
  const classBorder = CLASS_COLORS;

  const [hoveredNode, setHoveredNode] = useState(null);
  const [tipPos, setTipPos]           = useState({ x: 0, y: 0 });
  const [zoom, setZoom]               = useState(1);
  const containerRef = useRef(null);

  const { map, svgW, svgH } = useMemo(
    () => computeLayout(data.nodes),
    [data.nodes]
  );

  useEffect(() => {
    if (!containerRef.current || !svgW) return;
    const fitZoom = +Math.min(1, (containerRef.current.clientWidth - 16) / svgW).toFixed(2);
    setZoom(fitZoom);
  }, [svgW]);

  const zoomIn  = () => setZoom((z) => +Math.min(z + 0.15, 2.5).toFixed(2));
  const zoomOut = () => setZoom((z) => +Math.max(z - 0.15, 0.2).toFixed(2));
  const zoomFit = () => {
    if (!containerRef.current || !svgW) return;
    setZoom(+Math.min(1, (containerRef.current.clientWidth - 16) / svgW).toFixed(2));
  };

  const pathIds = useMemo(
    () => new Set((data.path ?? []).map((p) => p.node_id)),
    [data.path]
  );

  const pathEdges = useMemo(() => {
    const s    = new Set();
    const path = data.path ?? [];
    for (let i = 0; i < path.length - 1; i++) {
      s.add(`${path[i].node_id}-${path[i + 1].node_id}`);
    }
    return s;
  }, [data.path]);

  const edges = useMemo(() => {
    const list = [];
    Object.values(map).forEach((n) => {
      if (n.leaf) return;
      [[n.left_child, "yes"], [n.right_child, "no"]].forEach(([cid, lbl]) => {
        list.push({
          from: n, to: map[cid], lbl,
          act: pathEdges.has(`${n.node_id}-${cid}`),
        });
      });
    });
    return list;
  }, [map, pathEdges]);

  const pathSteps = (data.path ?? []).filter((p) => !p.leaf);
  const leafStep  = (data.path ?? []).find((p) => p.leaf);

  const onMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setTipPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  return (
    <div style={{
      background: T.bg,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif',
      display: "flex", flexDirection: "column",
      minHeight: "100vh",
    }}>

      {/* ── Header ── */}
      <div style={{
        background: T.surface,
        borderBottom: `0.5px solid ${T.border}`,
        padding: "12px 20px",
        display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
      }}>
        <span style={{ fontSize: 15, fontWeight: 600, color: T.text, letterSpacing: "-0.3px" }}>
          mlviz
        </span>
        <Pill T={T}>{(data.model_type ?? "decision tree").replace(/_/g, " ")}</Pill>
        <Pill T={T}>
          {data.nodes.length} nodes · {data.nodes.filter((n) => n.leaf).length} leaves
        </Pill>
        <div style={{ marginLeft: "auto", display: "flex", gap: 16, alignItems: "center" }}>
          {data.classes.map((c, idx) => (
            <div key={idx} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: T.textSecondary }}>
              <span style={{ width: 7, height: 7, borderRadius: "50%", background: classDot[idx] ?? T.textSecondary, display: "inline-block" }} />
              class {c}
            </div>
          ))}
          {/* ── Zoom controls ── */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <button
              onClick={zoomOut}
              style={{
                background: "none", border: `0.5px solid ${T.border}`, borderRadius: 6,
                padding: "3px 9px", cursor: "pointer", fontSize: 15,
                color: T.textSecondary, lineHeight: 1,
              }}
              aria-label="Zoom out"
            >−</button>
            <span style={{
              fontSize: 11, color: T.textSecondary,
              minWidth: 38, textAlign: "center",
              fontFamily: '"SF Mono", Menlo, monospace',
            }}>
              {Math.round(zoom * 100)}%
            </span>
            <button
              onClick={zoomIn}
              style={{
                background: "none", border: `0.5px solid ${T.border}`, borderRadius: 6,
                padding: "3px 9px", cursor: "pointer", fontSize: 15,
                color: T.textSecondary, lineHeight: 1,
              }}
              aria-label="Zoom in"
            >+</button>
            <button
              onClick={zoomFit}
              style={{
                background: "none", border: `0.5px solid ${T.border}`, borderRadius: 6,
                padding: "3px 8px", cursor: "pointer", fontSize: 11,
                color: T.textSecondary, letterSpacing: "0.1px",
              }}
              aria-label="Fit to screen"
            >fit</button>
          </div>

          <button
            onClick={toggleDark}
            style={{
              background: "none",
              border: `0.5px solid ${T.border}`,
              borderRadius: 8,
              padding: "4px 10px",
              cursor: "pointer",
              fontSize: 11,
              color: T.textSecondary,
              letterSpacing: "0.1px",
              marginLeft: 12,
            }}
            aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
          >
            {dark ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </div>

      {/* ── Tree canvas ── */}
      <div
        ref={containerRef}
        style={{ flex: 1, overflow: "auto", position: "relative", padding: "4px 0" }}
        onMouseMove={onMouseMove}
        onMouseLeave={() => setHoveredNode(null)}
      >
        <svg
          width={Math.round(svgW * zoom)}
          height={Math.round(svgH * zoom)}
          viewBox={`0 0 ${svgW} ${svgH}`}
          style={{ display: "block" }}
        >

          {/* Edges */}
          {edges.map((e, i) => {
            const midY = (e.from.py + NH + e.to.py) / 2;
            const lx   = (e.from.cx + e.to.cx) / 2;
            return (
              <g key={`${e.from.node_id}-${e.to.node_id}`}>
                <path
                  d={`M${e.from.cx},${e.from.py + NH} C${e.from.cx},${midY} ${e.to.cx},${midY} ${e.to.cx},${e.to.py}`}
                  fill="none"
                  stroke={e.act ? T.blue : T.border}
                  strokeWidth={e.act ? 2 : 1.5}
                />
                <text
                  x={lx} y={midY - 4}
                  textAnchor="middle" fontSize={9}
                  fill={e.act ? T.blue : T.textTertiary}
                  fontFamily="-apple-system, sans-serif"
                >
                  {e.lbl}
                </text>
              </g>
            );
          })}

          {/* Nodes */}
          {Object.values(map).map((n) => {
            const act = pathIds.has(n.node_id);
            const hov = hoveredNode?.node_id === n.node_id;
            const p   = n.leaf ? n.prediction : null;

            const fill   = n.leaf ? (act ? (classFill[p]   ?? T.surface)  : T.surface) : (act ? T.blueFill : T.surface);
            const stroke = n.leaf ? (act ? (classBorder[p] ?? T.border)   : hov ? T.borderHover : T.border)
                                  : (act ? T.blue : hov ? T.borderHover : T.border);
            const sw = act ? 1.5 : 1;

            return (
              <g
                key={n.node_id}
                style={{ cursor: "default" }}
                onMouseEnter={() => setHoveredNode(n)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <rect
                  x={n.px} y={n.py} width={NW} height={NH} rx={11}
                  fill={fill} stroke={stroke} strokeWidth={sw}
                />
                {n.leaf ? (
                  <>
                    <circle cx={n.px + 18} cy={n.cy} r={4.5} fill={classDot[p] ?? T.textSecondary} />
                    <text
                      x={n.px + 31} y={n.cy + 4}
                      fontSize={12} fontWeight="500"
                      fill={act ? T.text : T.textSecondary}
                      fontFamily="-apple-system, sans-serif"
                    >
                      {data.classes[p]}
                    </text>
                  </>
                ) : (
                  <text
                    x={n.cx} y={n.cy + 4}
                    textAnchor="middle"
                    fontSize={12} fontWeight={act ? "500" : "400"}
                    fill={act ? T.blue : T.text}
                    fontFamily='"SF Mono", Menlo, "Courier New", monospace'
                  >
                    {clampLabel(`${sf(n.feature)} ≤ ${n.threshold}`)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* ── Floating tooltip ── */}
        {hoveredNode && (
          <div style={{
            position: "absolute",
            left: Math.min(tipPos.x + 18, svgW * zoom - 200),
            top: Math.max(tipPos.y - 140, 8),
            pointerEvents: "none",
            zIndex: 20,
          }}>
            <Tooltip node={hoveredNode} map={map} classes={data.classes} T={T} classDot={classDot} classFill={classFill} />
          </div>
        )}
      </div>

      {/* ── Path bar (only rendered when a query point was passed) ── */}
      {data.path && (
        <div style={{
          background: T.surface,
          borderTop: `0.5px solid ${T.border}`,
          padding: "10px 20px",
          display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
        }}>
          <span style={{
            fontSize: 10, color: T.textTertiary,
            textTransform: "uppercase", letterSpacing: "0.6px", marginRight: 4,
          }}>
            path
          </span>

          {pathSteps.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 7 }}>
              {i > 0 && <span style={{ color: T.textTertiary, fontSize: 12 }}>→</span>}
              <span style={{ fontSize: 12, fontFamily: '"SF Mono", Menlo, monospace', color: T.text }}>
                {sf(s.feature)} ≤ {s.threshold}
              </span>
              <span style={{
                fontSize: 12, fontFamily: '"SF Mono", Menlo, monospace',
                color: s.went_left ? T.green : T.red,
                fontWeight: 500,
              }}>
                {s.value} → {s.went_left ? "yes" : "no"}
              </span>
            </div>
          ))}

          {leafStep && (
            <>
              <span style={{ color: T.textTertiary, fontSize: 12 }}>→</span>
              <span style={{
                fontSize: 12, fontWeight: 500,
                background: classFill[leafStep.prediction]   ?? T.surfaceSecondary,
                border: `0.5px solid ${classBorder[leafStep.prediction] ?? T.border}`,
                borderRadius: 7, padding: "3px 11px",
                color: classBorder[leafStep.prediction] ?? T.text,
              }}>
                {data.classes[leafStep.prediction]}
              </span>
            </>
          )}
        </div>
      )}

    </div>
  );
}
