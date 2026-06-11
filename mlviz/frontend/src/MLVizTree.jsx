import { useState, useEffect, useMemo, useRef } from "react";
import { motion, AnimatePresence, useReducedMotion } from "framer-motion";
import {
  getTheme, getClassColors, sf, clampLabel, FONT_UI, FONT_MONO,
} from "./theme";

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
function TRow({ label, value, T }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "2.5px 0", gap: 18 }}>
      <span style={{ fontSize: 11, color: T.textSecondary }}>{label}</span>
      <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO }}>
        {value}
      </span>
    </div>
  );
}

function Tooltip({ node, map, classes, T, classDot }) {
  if (!node) return null;
  const p = node.prediction;
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: 8,
      padding: "12px 14px",
      minWidth: 190,
      boxShadow: "0 4px 24px rgba(0, 0, 0, 0.25)",
    }}>
      <div style={{
        fontSize: 10, color: T.textTertiary, letterSpacing: "0.7px",
        textTransform: "uppercase", marginBottom: 6, fontWeight: 500,
      }}>
        {node.leaf ? "leaf node" : "split node"}
      </div>

      {node.leaf ? (
        <>
          <div style={{
            fontSize: 13, fontWeight: 600, color: classDot[p] ?? T.textSecondary,
            marginBottom: 10, display: "flex", alignItems: "center", gap: 6,
          }}>
            <span style={{
              width: 8, height: 8, borderRadius: 2,
              background: classDot[p] ?? T.textSecondary,
              display: "inline-block", flexShrink: 0,
            }} />
            {classes[p]}
          </div>
          <TRow label="gini" value={node.gini.toFixed(3)} T={T} />
          <TRow label="samples" value={node.n_samples} T={T} />
          <div style={{ height: 1, background: T.separator, margin: "8px 0" }} />
          {classes.map((c, idx) => (
            <div key={c} style={{ display: "flex", justifyContent: "space-between", padding: "2.5px 0", gap: 18 }}>
              <span style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: T.textSecondary }}>
                <span style={{
                  width: 6, height: 6, borderRadius: 2,
                  background: classDot[idx] ?? T.textSecondary,
                  display: "inline-block",
                }} />
                {c}
              </span>
              <span style={{ fontSize: 11, fontFamily: FONT_MONO, color: T.text }}>
                {node.counts?.[idx] ?? 0}
              </span>
            </div>
          ))}
        </>
      ) : (
        <>
          <div style={{
            fontSize: 13, fontWeight: 500, color: T.text, marginBottom: 10,
            fontFamily: FONT_MONO,
          }}>
            {sf(node.feature)} ≤ {node.threshold}
          </div>
          <TRow label="gini" value={node.gini.toFixed(3)} T={T} />
          <TRow label="samples" value={node.n_samples} T={T} />
          <div style={{ height: 1, background: T.separator, margin: "8px 0" }} />
          <TRow label="left branch" value={map[node.left_child]?.n_samples} T={T} />
          <TRow label="right branch" value={map[node.right_child]?.n_samples} T={T} />
        </>
      )}
    </div>
  );
}

function ZoomButton({ onClick, label, children, T, wide }) {
  return (
    <button
      onClick={onClick}
      className="zoom-btn"
      style={{
        background: "transparent", border: "none",
        padding: wide ? "5px 10px" : "5px 11px",
        cursor: "pointer", fontSize: wide ? 11 : 14,
        color: T.textSecondary, lineHeight: 1,
        fontFamily: wide ? FONT_UI : FONT_MONO,
      }}
      aria-label={label}
    >
      {children}
    </button>
  );
}

// ─── Single-tree canvas renderer ──────────────────────────────────────────────
// Model-agnostic: renders any sklearn-shaped tree (nodes + optional path).
// Headers, panels, and ensemble chrome are owned by the view components.
export default function MLVizTree({ nodes, path, classes, dark }) {
  const T = getTheme(dark);
  const classDot = getClassColors(dark);
  const classFill = T.classFill;

  const reduceMotion = useReducedMotion();
  const tooltipTransition = reduceMotion
    ? { duration: 0 }
    : { duration: 0.15, ease: [0.23, 1, 0.32, 1] };
  const pathBarTransition = reduceMotion
    ? { duration: 0 }
    : { duration: 0.2, ease: [0.23, 1, 0.32, 1] };

  const [hoveredNode, setHoveredNode] = useState(null);
  const [zoom, setZoom] = useState(1);
  const containerRef = useRef(null);
  const tipRef = useRef({ x: 0, y: 0 });
  const tipDivRef = useRef(null);

  const { map, svgW, svgH } = useMemo(() => computeLayout(nodes), [nodes]);

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
    () => new Set((path ?? []).map((p) => p.node_id)),
    [path]
  );

  const pathEdges = useMemo(() => {
    const s = new Set();
    const steps = path ?? [];
    for (let i = 0; i < steps.length - 1; i++) {
      s.add(`${steps[i].node_id}-${steps[i + 1].node_id}`);
    }
    return s;
  }, [path]);

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

  const pathSteps = (path ?? []).filter((p) => !p.leaf);
  const leafStep  = (path ?? []).find((p) => p.leaf);

  const onMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    tipRef.current = { x, y };
    if (tipDivRef.current) {
      tipDivRef.current.style.left = `${Math.min(x + 18, rect.width - 210)}px`;
      tipDivRef.current.style.top = `${Math.max(y - 140, 8)}px`;
    }
  };

  return (
    <div style={{
      flex: 1, minHeight: 0, minWidth: 0,
      display: "flex", flexDirection: "column",
      fontFamily: FONT_UI,
    }}>

      {/* ── Tree canvas ── */}
      <div
        ref={containerRef}
        style={{
          flex: 1, minHeight: 0, overflow: "auto", position: "relative",
          backgroundColor: T.canvas,
          backgroundImage: `radial-gradient(circle, ${T.gridDot} 1px, transparent 1px)`,
          backgroundSize: "24px 24px",
        }}
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
          {edges.map((e) => {
            const midY = (e.from.py + NH + e.to.py) / 2;
            const lx = (e.from.cx + e.to.cx) / 2;
            return (
              <g key={`${e.from.node_id}-${e.to.node_id}`}>
                <path
                  d={`M${e.from.cx},${e.from.py + NH} C${e.from.cx},${midY} ${e.to.cx},${midY} ${e.to.cx},${e.to.py}`}
                  fill="none"
                  stroke={e.act ? T.accent : T.nodeBorder}
                  strokeWidth={e.act ? 2 : 1.25}
                  style={{ transition: "stroke 180ms ease-out" }}
                />
                <text
                  x={lx} y={midY - 5}
                  textAnchor="middle" fontSize={9}
                  fill={e.act ? T.accent : T.textTertiary}
                  fontFamily={FONT_MONO}
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
            const p = n.leaf ? n.prediction : null;

            const fill = n.leaf
              ? (act ? (classFill[p] ?? T.nodeFill) : T.nodeFill)
              : (act ? T.accentFill : T.nodeFill);
            const stroke = n.leaf
              ? (act ? (classDot[p] ?? T.nodeBorder) : hov ? T.nodeBorderHover : T.nodeBorder)
              : (act ? T.accent : hov ? T.nodeBorderHover : T.nodeBorder);
            const sw = act ? 1.5 : 1;

            return (
              <g
                key={n.node_id}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHoveredNode(n)}
                onMouseLeave={() => setHoveredNode(null)}
              >
                <rect
                  x={n.px} y={n.py} width={NW} height={NH} rx={8}
                  fill={fill} stroke={stroke} strokeWidth={sw}
                  style={{ transition: "fill 180ms ease-out, stroke 180ms ease-out" }}
                />
                {n.leaf ? (
                  <>
                    <rect
                      x={n.px + 14} y={n.cy - 4} width={8} height={8} rx={2}
                      fill={classDot[p] ?? T.textSecondary}
                    />
                    <text
                      x={n.px + 31} y={n.cy + 4}
                      fontSize={12} fontWeight="500"
                      fill={act ? T.text : T.textSecondary}
                      fontFamily={FONT_UI}
                    >
                      {classes[p]}
                    </text>
                  </>
                ) : (
                  <text
                    x={n.cx} y={n.cy + 4}
                    textAnchor="middle"
                    fontSize={12} fontWeight={act ? "500" : "400"}
                    fill={act ? T.accent : T.text}
                    fontFamily={FONT_MONO}
                  >
                    {clampLabel(`${sf(n.feature)} ≤ ${n.threshold}`)}
                  </text>
                )}
              </g>
            );
          })}
        </svg>

        {/* ── Zoom controls (floating, bottom-right) ── */}
        <div style={{
          position: "absolute", right: 14, bottom: 14,
          display: "flex", alignItems: "center",
          background: T.surface,
          border: `1px solid ${T.border}`,
          borderRadius: 7,
          boxShadow: "0 2px 12px rgba(0, 0, 0, 0.18)",
          zIndex: 10,
        }}>
          <ZoomButton onClick={zoomOut} label="Zoom out" T={T}>−</ZoomButton>
          <span style={{
            fontSize: 11, color: T.textSecondary,
            minWidth: 42, textAlign: "center",
            fontFamily: FONT_MONO,
            borderLeft: `1px solid ${T.separator}`,
            borderRight: `1px solid ${T.separator}`,
            padding: "5px 0",
          }}>
            {Math.round(zoom * 100)}%
          </span>
          <ZoomButton onClick={zoomIn} label="Zoom in" T={T}>+</ZoomButton>
          <span style={{ width: 1, height: 16, background: T.separator }} />
          <ZoomButton onClick={zoomFit} label="Fit to screen" T={T} wide>fit</ZoomButton>
        </div>

        {/* ── Floating tooltip ── */}
        <AnimatePresence>
          {hoveredNode && (
            <motion.div
              ref={tipDivRef}
              key={hoveredNode.node_id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={tooltipTransition}
              style={{
                position: "absolute",
                left: tipRef.current.x + 18,
                top: Math.max(tipRef.current.y - 140, 8),
                pointerEvents: "none",
                zIndex: 20,
                transformOrigin: "top left",
              }}
            >
              <Tooltip node={hoveredNode} map={map} classes={classes} T={T} classDot={classDot} />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* ── Path bar (only rendered when a query point was passed) ── */}
      {path && (
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={pathBarTransition}
          style={{
            background: T.surface,
            borderTop: `1px solid ${T.border}`,
            padding: "9px 16px",
            display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap",
            flexShrink: 0,
          }}
        >
          <span style={{
            fontSize: 10, color: T.textTertiary, fontWeight: 500,
            textTransform: "uppercase", letterSpacing: "0.6px", marginRight: 4,
          }}>
            path
          </span>

          {pathSteps.map((s, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 7 }}>
              {i > 0 && <span style={{ color: T.textTertiary, fontSize: 12 }}>→</span>}
              <span style={{ fontSize: 12, fontFamily: FONT_MONO, color: T.text }}>
                {sf(s.feature)} ≤ {s.threshold}
              </span>
              <span style={{
                fontSize: 12, fontFamily: FONT_MONO,
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
                background: classFill[leafStep.prediction] ?? T.surfaceSecondary,
                border: `1px solid ${classDot[leafStep.prediction] ?? T.border}`,
                borderRadius: 5, padding: "2px 10px",
                color: classDot[leafStep.prediction] ?? T.text,
              }}>
                {classes[leafStep.prediction]}
              </span>
            </>
          )}
        </motion.div>
      )}
    </div>
  );
}
