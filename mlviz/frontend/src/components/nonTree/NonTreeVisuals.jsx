import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Panel from "../forest/Panel";
import { getTheme, getClassColors, FONT_MONO, FONT_UI } from "../../theme";

const WORLD_SCALE = 100;
const GRID_SIZE = 80;

function fmt(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(3);
  return String(value);
}

function extent(values) {
  const nums = values.filter((v) => Number.isFinite(v));
  if (!nums.length) return [0, 1];
  const min = Math.min(...nums);
  const max = Math.max(...nums);
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

function scale(value, fromMin, fromMax, toMin, toMax) {
  return toMin + ((value - fromMin) / (fromMax - fromMin)) * (toMax - toMin);
}

function hexToRgb(hex) {
  const c = hex.replace("#", "");
  return [parseInt(c.slice(0, 2), 16), parseInt(c.slice(2, 4), 16), parseInt(c.slice(4, 6), 16)];
}

function mixHex(a, b, t) {
  const ca = hexToRgb(a);
  const cb = hexToRgb(b);
  const m = ca.map((ch, i) => Math.round(ch + (cb[i] - ch) * t));
  return `rgb(${m[0]}, ${m[1]}, ${m[2]})`;
}

function worldToScreen(wx, wy, zoom, ox, oy, cw, ch) {
  return {
    sx: cw / 2 + (wx * WORLD_SCALE + ox) * zoom,
    sy: ch / 2 + (-wy * WORLD_SCALE + oy) * zoom,
  };
}

function computeAutoFit(points, cw, ch) {
  if (!points?.length) return { zoom: 1, ox: 0, oy: 0 };
  const wxMin = Math.min(...points.map((p) => p.wx));
  const wxMax = Math.max(...points.map((p) => p.wx));
  const wyMin = Math.min(...points.map((p) => p.wy));
  const wyMax = Math.max(...points.map((p) => p.wy));
  const dataW = wxMax - wxMin || 1;
  const dataH = wyMax - wyMin || 1;
  const zoom = Math.max(0.2, Math.min(6,
    Math.min(cw * 0.8 / (dataW * WORLD_SCALE), ch * 0.8 / (dataH * WORLD_SCALE))
  ));
  return {
    zoom,
    ox: -((wxMin + wxMax) / 2) * WORLD_SCALE,
    oy: ((wyMin + wyMax) / 2) * WORLD_SCALE,
  };
}

function useCanvasSetup(canvasRef) {
  const [size, setSize] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const w = el.clientWidth;
      const h = el.clientHeight;
      const dpr = window.devicePixelRatio || 1;
      el.width = Math.round(w * dpr);
      el.height = Math.round(h * dpr);
      setSize({ w, h });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return size;
}

function TooltipBox({ item, T }) {
  if (!item) return null;
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: 8,
      padding: "10px 12px",
      boxShadow: "0 4px 24px rgba(0, 0, 0, 0.25)",
      minWidth: 150,
    }}>
      {item.lines.map((line) => (
        <div key={line.label} style={{ display: "flex", justifyContent: "space-between", gap: 16, padding: "2px 0" }}>
          <span style={{ fontSize: 11, color: T.textTertiary }}>{line.label}</span>
          <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO }}>{line.value}</span>
        </div>
      ))}
    </div>
  );
}

function ChartFrame({ children, hover, setHover, dark }) {
  const T = getTheme(dark);
  const reduceMotion = useReducedMotion();
  const ref = useRef(null);
  const transition = reduceMotion ? { duration: 0 } : { duration: 0.15, ease: [0.23, 1, 0.32, 1] };
  return (
    <div
      ref={ref}
      style={{ position: "relative", overflowX: "auto" }}
      onMouseMove={(e) => {
        if (!hover || !ref.current) return;
        const rect = ref.current.getBoundingClientRect();
        setHover({ ...hover, x: e.clientX - rect.left + 16, y: e.clientY - rect.top + 16 });
      }}
      onMouseLeave={() => setHover(null)}
    >
      {children}
      <AnimatePresence>
        {hover && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={transition}
            style={{
              position: "absolute",
              left: Math.min(hover.x, 260),
              top: Math.max(8, hover.y - 80),
              zIndex: 20,
              pointerEvents: "none",
              transformOrigin: "top left",
            }}
          >
            <TooltipBox item={hover.item} T={T} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ── Unchanged sidebar chart components ───────────────────────────────────────

export function KnnVoteBarChart({ data, dark }) {
  const T = getTheme(dark);
  const votes = data.vote_distribution?.class_votes ?? [];
  if (!votes.length) {
    return (
      <Panel title="neighbor vote weights" dark={dark}>
        <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>No query sample supplied - vote chart is unavailable.</div>
      </Panel>
    );
  }
  const max = Math.max(...votes.map((v) => v.weight), 1);
  return (
    <Panel title="neighbor vote weights" dark={dark}>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {votes.map((vote) => {
          const winner = vote.class_index === data.vote_distribution.winning_class_index;
          return (
            <div key={vote.class_index}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 3 }}>
                <span style={{ fontSize: 11, color: winner ? T.accent : T.textSecondary }}>{String(vote.class_label)}</span>
                <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO }}>{fmt(vote.weight)}</span>
              </div>
              <svg width="100%" height="8" viewBox="0 0 100 8" preserveAspectRatio="none">
                <rect x="0" y="2" width="100" height="4" rx="2" fill={T.surfaceSecondary} />
                <rect
                  x="0" y="2"
                  width={(vote.weight / max) * 100}
                  height="4" rx="2"
                  fill={winner ? T.accent : T.textTertiary}
                  style={{ transition: "width 220ms cubic-bezier(0.23, 1, 0.32, 1), fill 180ms ease-out" }}
                />
              </svg>
            </div>
          );
        })}
      </div>
    </Panel>
  );
}

export function KnnRegressionTargetStrip({ data, dark }) {
  const T = getTheme(dark);
  const [hover, setHover] = useState(null);
  const neighbors = data.neighbors ?? [];
  if (!neighbors.length || data.prediction === null) {
    return (
      <Panel title="neighbor target strip" dark={dark}>
        <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>No query sample supplied - target strip is unavailable.</div>
      </Panel>
    );
  }
  const [min, max] = extent([...neighbors.map((n) => n.target_value), data.prediction.value]);
  const xFor = (v) => scale(v, min, max, 28, 252);
  return (
    <Panel title="neighbor target strip" dark={dark}>
      <ChartFrame hover={hover} setHover={setHover} dark={dark}>
        <svg width="280" height="76" viewBox="0 0 280 76">
          <line x1="28" y1="38" x2="252" y2="38" stroke={T.border} />
          <line x1={xFor(data.prediction.value)} y1="18" x2={xFor(data.prediction.value)} y2="58" stroke={T.accent} strokeWidth="1.5" />
          {neighbors.map((n) => (
            <circle
              key={n.training_index}
              cx={xFor(n.target_value)} cy="38" r="5"
              fill={T.green} stroke={T.surface} strokeWidth="1.5"
              style={{ transition: "fill 180ms ease-out, stroke 180ms ease-out" }}
              onMouseEnter={(e) => setHover({
                x: e.nativeEvent.offsetX + 16, y: e.nativeEvent.offsetY + 16,
                item: { lines: [
                  { label: "rank", value: n.rank },
                  { label: "target", value: fmt(n.target_value) },
                  { label: "distance", value: fmt(n.distance) },
                ] },
              })}
            />
          ))}
          <text x="28" y="70" fill={T.textTertiary} fontSize="10" fontFamily={FONT_MONO}>{fmt(min)}</text>
          <text x="252" y="70" textAnchor="end" fill={T.textTertiary} fontSize="10" fontFamily={FONT_MONO}>{fmt(max)}</text>
          <text x={xFor(data.prediction.value)} y="13" textAnchor="middle" fill={T.accent} fontSize="10" fontFamily={FONT_MONO}>pred {fmt(data.prediction.value)}</text>
        </svg>
      </ChartFrame>
    </Panel>
  );
}

export function SvmDecisionScoreChart({ data, dark }) {
  const T = getTheme(dark);
  const scores = data.prediction?.decision_scores ?? [];
  if (!scores.length) {
    return (
      <Panel title="decision scores" dark={dark}>
        <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>Decision score bars are unavailable for this SVC output shape.</div>
      </Panel>
    );
  }
  const maxAbs = Math.max(...scores.map((s) => Math.abs(s.score)), 1);
  const zeroX = 132;
  return (
    <Panel title="decision scores" dark={dark}>
      <svg width="280" height={scores.length * 28 + 18} viewBox={`0 0 280 ${scores.length * 28 + 18}`}>
        <line x1={zeroX} y1="2" x2={zeroX} y2={scores.length * 28 + 10} stroke={T.border} />
        {scores.map((s, i) => {
          const y = i * 28 + 9;
          const w = (Math.abs(s.score) / maxAbs) * 110;
          const x = s.score >= 0 ? zeroX : zeroX - w;
          return (
            <g key={s.class_index}>
              <text x="0" y={y + 10} fill={s.is_winner ? T.accent : T.textSecondary} fontSize="11">{String(s.class_label)}</text>
              <rect x={x} y={y} width={w} height="12" rx="3" fill={s.is_winner ? T.accent : T.textTertiary}
                style={{ transition: "width 220ms cubic-bezier(0.23, 1, 0.32, 1), fill 180ms ease-out" }} />
              <text x="270" y={y + 10} textAnchor="end" fill={T.text} fontSize="11" fontFamily={FONT_MONO}>{fmt(s.score)}</text>
            </g>
          );
        })}
      </svg>
    </Panel>
  );
}

export function SvmRegressionValueStrip({ data, dark }) {
  const T = getTheme(dark);
  const [hover, setHover] = useState(null);
  const range = data.support_vector_target_range;
  const prediction = data.prediction?.value;
  if (!range || prediction === undefined) {
    return (
      <Panel title="support target range" dark={dark}>
        <div style={{ fontSize: 12, color: T.textTertiary, lineHeight: 1.5 }}>No query sample supplied - prediction marker is unavailable.</div>
      </Panel>
    );
  }
  const [min, max] = extent([range.min, range.max, prediction]);
  const xFor = (v) => scale(v, min, max, 28, 252);
  return (
    <Panel title="support target range" dark={dark}>
      <ChartFrame hover={hover} setHover={setHover} dark={dark}>
        <svg width="280" height="76" viewBox="0 0 280 76">
          <line x1="28" y1="38" x2="252" y2="38" stroke={T.border} />
          <rect x={xFor(range.min)} y="29" width={Math.max(2, xFor(range.max) - xFor(range.min))} height="18" rx="5" fill={T.accentFill} stroke={T.accent} strokeWidth="1" />
          <circle
            cx={xFor(prediction)} cy="38" r="6"
            fill={T.accent} stroke={T.surface} strokeWidth="1.5"
            onMouseEnter={(e) => setHover({
              x: e.nativeEvent.offsetX + 16, y: e.nativeEvent.offsetY + 16,
              item: { lines: [
                { label: "prediction", value: fmt(prediction) },
                { label: "sv min", value: fmt(range.min) },
                { label: "sv max", value: fmt(range.max) },
              ] },
            })}
          />
          <text x={xFor(range.min)} y="64" textAnchor="middle" fill={T.textTertiary} fontSize="10" fontFamily={FONT_MONO}>min {fmt(range.min)}</text>
          <text x={xFor(range.max)} y="64" textAnchor="middle" fill={T.textTertiary} fontSize="10" fontFamily={FONT_MONO}>max {fmt(range.max)}</text>
        </svg>
      </ChartFrame>
    </Panel>
  );
}

// ── Shared canvas helpers ─────────────────────────────────────────────────────

function drawRegionTiles(ctx, regions, T, ws, zoom) {
  for (const r of regions) {
    const { sx: cx, sy: cy } = ws(r.wx, r.wy);
    const pw = r.cell_w * WORLD_SCALE * zoom;
    const ph = r.cell_h * WORLD_SCALE * zoom;
    ctx.fillStyle = T.classFill[r.class_index] ?? T.classFill[0];
    ctx.fillRect(cx - pw / 2, cy - ph / 2, pw, ph);
  }
}

function drawGridBoundaryCanvas(ctx, regions, field, ws) {
  const G = GRID_SIZE;
  ctx.beginPath();
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    const col = i % G;
    const row = Math.floor(i / G);
    if (col < G - 1) {
      const right = regions[i + 1];
      if (right && right[field] !== r[field]) {
        const midWx = (r.wx + right.wx) / 2;
        const { sx, sy: sy1 } = ws(midWx, r.wy + r.cell_h / 2);
        const { sy: sy2 } = ws(midWx, r.wy - r.cell_h / 2);
        ctx.moveTo(sx, sy1);
        ctx.lineTo(sx, sy2);
      }
    }
    if (row < G - 1) {
      const above = regions[i + G];
      if (above && above[field] !== r[field]) {
        const midWy = (r.wy + above.wy) / 2;
        const { sx: sx1, sy } = ws(r.wx - r.cell_w / 2, midWy);
        const { sx: sx2 } = ws(r.wx + r.cell_w / 2, midWy);
        ctx.moveTo(sx1, sy);
        ctx.lineTo(sx2, sy);
      }
    }
  }
  ctx.stroke();
}

function drawQueryX(ctx, wx, wy, T, ws) {
  const { sx, sy } = ws(wx, wy);
  ctx.font = `bold 22px ${FONT_MONO}`;
  ctx.fillStyle = T.accent;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("×", sx, sy);
}

function CanvasTooltip({ hover, T, reduceMotion }) {
  const transition = reduceMotion ? { duration: 0 } : { duration: 0.12, ease: [0.23, 1, 0.32, 1] };
  return (
    <AnimatePresence>
      {hover && (
        <motion.div
          key="tip"
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          transition={transition}
          style={{
            position: "absolute",
            left: hover.sx + 12,
            top: Math.max(8, hover.sy - 60),
            zIndex: 20,
            pointerEvents: "none",
            transformOrigin: "top left",
          }}
        >
          <TooltipBox item={hover.item} T={T} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function useCanvasInteraction(canvasRef, exploreMode) {
  const [transform, setTransform] = useState({ zoom: 1, ox: 0, oy: 0 });
  const transformRef = useRef(transform);
  const dragRef = useRef(null);

  useEffect(() => { transformRef.current = transform; }, [transform]);

  // Register non-passive wheel listener only when explore mode is active.
  // When inactive, no listener is registered so the page scrolls normally.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el || !exploreMode) return;
    const onWheel = (e) => {
      e.preventDefault();
      const { zoom, ox, oy } = transformRef.current;
      const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
      const newZoom = Math.max(0.2, Math.min(6, zoom * factor));
      const rect = el.getBoundingClientRect();
      const k = 1 / newZoom - 1 / zoom;
      setTransform({
        zoom: newZoom,
        ox: ox + (e.clientX - rect.left - el.clientWidth / 2) * k,
        oy: oy + (e.clientY - rect.top - el.clientHeight / 2) * k,
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [canvasRef, exploreMode]);

  const onMouseDown = (e) => {
    if (!exploreMode || e.button !== 0) return;
    const { ox, oy, zoom } = transformRef.current;
    dragRef.current = { startX: e.clientX, startY: e.clientY, startOx: ox, startOy: oy, startZoom: zoom };
  };

  const onMouseUp = () => { dragRef.current = null; };

  return { transform, setTransform, dragRef, onMouseDown, onMouseUp };
}

// ── KnnProjectionPlot ─────────────────────────────────────────────────────────

export function KnnProjectionPlot({ data, dark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const canvasRef = useRef(null);
  const canvasSize = useCanvasSetup(canvasRef);
  const reduceMotion = useReducedMotion();
  const [exploreMode, setExploreMode] = useState(false);
  const { transform, setTransform, dragRef, onMouseDown, onMouseUp } = useCanvasInteraction(canvasRef, exploreMode);
  const { zoom, ox, oy } = transform;
  const [hover, setHover] = useState(null);

  const projection = data.projection;
  const taskType = data.task_type ?? "classification";

  // Clear drag + hover when leaving explore mode
  useEffect(() => {
    if (!exploreMode) {
      dragRef.current = null;
      setHover(null);
    }
  }, [exploreMode, dragRef]);

  // Escape key exits explore mode
  useEffect(() => {
    if (!exploreMode) return;
    const fn = (e) => { if (e.key === "Escape") setExploreMode(false); };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [exploreMode]);

  // Auto-fit once on first valid canvas size
  const hasFitted = useRef(false);
  useEffect(() => {
    if (hasFitted.current || !canvasSize.w || !canvasSize.h || !projection?.points?.length) return;
    hasFitted.current = true;
    setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
  }, [canvasSize, projection, setTransform]);

  const neighborSet = useMemo(
    () => new Set((data.neighbors ?? []).map((n) => n.training_index)),
    [data.neighbors]
  );
  const neighborByIdx = useMemo(
    () => new Map((data.neighbors ?? []).map((n) => [n.training_index, n])),
    [data.neighbors]
  );

  const toWorldY = useMemo(() => {
    if (taskType !== "regression" || !projection) return null;
    const ys = [...projection.points.map((p) => p.label_or_target)];
    if (data.prediction?.value != null) ys.push(data.prediction.value);
    const [mn, mx] = extent(ys);
    const mid = (mn + mx) / 2;
    const half = Math.max((mx - mn) / 2, 1e-6);
    return (v) => ((v - mid) / half) * 3;
  }, [taskType, projection, data.prediction]);

  const targetRange = useMemo(() => {
    if (taskType !== "regression" || !projection) return null;
    return extent(projection.points.map((p) => p.label_or_target));
  }, [taskType, projection]);

  useEffect(() => {
    if (!canvasSize.w || !canvasSize.h || !projection) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h);

    const ws = (wx, wy) => worldToScreen(wx, wy, zoom, ox, oy, canvasSize.w, canvasSize.h);
    const ptWy = (p) => (taskType === "regression" && toWorldY ? toWorldY(p.label_or_target) : p.wy);

    if (taskType === "classification" && projection.regions) {
      drawRegionTiles(ctx, projection.regions, T, ws, zoom);
      ctx.lineWidth = 1;
      ctx.strokeStyle = T.text;
      ctx.globalAlpha = 0.28;
      drawGridBoundaryCanvas(ctx, projection.regions, "class_index", ws);
      ctx.globalAlpha = 1;
    }

    if (projection.query_point && projection.neighbor_indices?.length) {
      const { sx: qsx, sy: qsy } = ws(projection.query_point.wx, projection.query_point.wy);
      ctx.lineWidth = 1;
      ctx.strokeStyle = T.accent;
      ctx.globalAlpha = 0.35;
      ctx.beginPath();
      for (const idx of projection.neighbor_indices) {
        const pt = projection.points[idx];
        if (!pt) continue;
        const { sx, sy } = ws(pt.wx, ptWy(pt));
        ctx.moveTo(qsx, qsy);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    for (const pt of projection.points) {
      const isNeighbor = neighborSet.has(pt.training_index);
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      const r = isNeighbor ? 5 : 3.5;
      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }
      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isNeighbor ? 1.75 : 1;
      ctx.strokeStyle = isNeighbor ? T.accent : T.surface;
      ctx.stroke();
    }

    if (projection.query_point) {
      const qWy = taskType === "regression" && toWorldY && data.prediction?.value != null
        ? toWorldY(data.prediction.value)
        : projection.query_point.wy;
      drawQueryX(ctx, projection.query_point.wx, qWy, T, ws);
    }
  }, [canvasSize, transform, dark, data]);

  const onMouseMove = (e) => {
    if (!projection || !exploreMode) return;
    if (dragRef.current) {
      const { startX, startY, startOx, startOy, startZoom } = dragRef.current;
      setTransform((prev) => ({
        ...prev,
        ox: startOx + (e.clientX - startX) / startZoom,
        oy: startOy + (e.clientY - startY) / startZoom,
      }));
      setHover(null);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    let best = null, bestDist = 12;
    for (const pt of projection.points) {
      const pwy = taskType === "regression" && toWorldY ? toWorldY(pt.label_or_target) : pt.wy;
      const { sx: px, sy: py } = worldToScreen(pt.wx, pwy, zoom, ox, oy, canvasSize.w, canvasSize.h);
      const d = Math.hypot(px - sx, py - sy);
      if (d < bestDist) { bestDist = d; best = { pt, sx, sy }; }
    }
    if (best) {
      const n = neighborByIdx.get(best.pt.training_index);
      setHover({
        sx: best.sx, sy: best.sy,
        item: { lines: [
          { label: "row", value: best.pt.training_index },
          { label: taskType === "regression" ? "target" : "class", value: fmt(best.pt.label_or_target) },
          { label: "distance", value: n ? fmt(n.distance) : "n/a" },
        ] },
      });
    } else {
      setHover(null);
    }
  };

  if (!projection) return null;

  return (
    <div
      style={{ flex: 1, position: "relative", overflow: "hidden" }}
      onMouseMove={onMouseMove}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={() => { dragRef.current = null; setHover(null); }}
      onDoubleClick={() => {
        if (!exploreMode || !projection?.points?.length || !canvasSize.w) return;
        setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", cursor: exploreMode ? "crosshair" : "default" }}
      />
      <div style={{ position: "absolute", top: 8, left: 10, fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", pointerEvents: "none" }}>
        KNN decision map · PCA projection
      </div>
      <button
        onClick={() => setExploreMode((v) => !v)}
        style={{
          position: "absolute",
          bottom: 10,
          left: 10,
          padding: "4px 10px",
          background: exploreMode ? T.accent : T.surface,
          border: `1px solid ${exploreMode ? T.accent : T.border}`,
          borderRadius: 6,
          fontSize: 10,
          fontWeight: 500,
          textTransform: "uppercase",
          letterSpacing: "0.7px",
          color: exploreMode ? "#fff" : T.textTertiary,
          cursor: "pointer",
          fontFamily: FONT_UI,
          transition: "background 150ms ease-out, color 150ms ease-out, border-color 150ms ease-out",
        }}
      >
        {exploreMode ? "exit" : "explore"}
      </button>
      {exploreMode ? (
        <div style={{ position: "absolute", bottom: 10, right: 10, fontSize: 10, color: T.textTertiary, pointerEvents: "none" }}>
          scroll to zoom · drag to pan · double-click to reset · esc to exit
        </div>
      ) : (
        <div style={{ position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)", fontSize: 10, color: T.textTertiary, pointerEvents: "none", whiteSpace: "nowrap" }}>
          click explore to interact
        </div>
      )}
      <CanvasTooltip hover={hover} T={T} reduceMotion={reduceMotion} />
    </div>
  );
}

// ── SvmProjectionPlot ─────────────────────────────────────────────────────────

export function SvmProjectionPlot({ data, dark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const canvasRef = useRef(null);
  const canvasSize = useCanvasSetup(canvasRef);
  const reduceMotion = useReducedMotion();
  const [exploreMode, setExploreMode] = useState(false);
  const { transform, setTransform, dragRef, onMouseDown, onMouseUp } = useCanvasInteraction(canvasRef, exploreMode);
  const { zoom, ox, oy } = transform;
  const [hover, setHover] = useState(null);

  const projection = data.projection;
  const taskType = data.task_type ?? "classification";

  useEffect(() => {
    if (!exploreMode) {
      dragRef.current = null;
      setHover(null);
    }
  }, [exploreMode, dragRef]);

  useEffect(() => {
    if (!exploreMode) return;
    const fn = (e) => { if (e.key === "Escape") setExploreMode(false); };
    window.addEventListener("keydown", fn);
    return () => window.removeEventListener("keydown", fn);
  }, [exploreMode]);

  const hasFitted = useRef(false);
  useEffect(() => {
    if (hasFitted.current || !canvasSize.w || !canvasSize.h || !projection?.points?.length) return;
    hasFitted.current = true;
    setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
  }, [canvasSize, projection, setTransform]);

  // SVR: normalize raw target / prediction values to world-Y space
  const toWorldY = useMemo(() => {
    if (taskType !== "regression" || !projection) return null;
    const ys = [...projection.points.map((p) => p.label_or_target)];
    if (projection.fit_line) ys.push(...projection.fit_line.map((p) => p.wy));
    if (data.prediction?.value != null) ys.push(data.prediction.value);
    const [mn, mx] = extent(ys);
    const mid = (mn + mx) / 2;
    const half = Math.max((mx - mn) / 2, 1e-6);
    return { fn: (v) => ((v - mid) / half) * 3, half };
  }, [taskType, projection, data.prediction]);

  const targetRange = useMemo(() => {
    if (taskType !== "regression" || !projection) return null;
    return extent(projection.points.map((p) => p.label_or_target));
  }, [taskType, projection]);

  // Draw
  useEffect(() => {
    if (!canvasSize.w || !canvasSize.h || !projection) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h);

    const ws = (wx, wy) => worldToScreen(wx, wy, zoom, ox, oy, canvasSize.w, canvasSize.h);
    const toWY = toWorldY ? toWorldY.fn : null;
    const ptWy = (p) => (taskType === "regression" && toWY ? toWY(p.label_or_target) : p.wy);

    if (taskType === "classification") {
      // 1. Region tiles
      if (projection.regions) {
        drawRegionTiles(ctx, projection.regions, T, ws, zoom);
        ctx.lineWidth = 1;
        ctx.strokeStyle = T.text;
        ctx.globalAlpha = 0.28;
        drawGridBoundaryCanvas(ctx, projection.regions, "class_index", ws);
        ctx.globalAlpha = 1;
      }

      // 2. Margin boundary and dashed margin lines (binary SVC only)
      if (projection.margin_regions) {
        // Boundary: where margin_sign changes (0 crossing)
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = T.text;
        ctx.globalAlpha = 0.7;
        const G = GRID_SIZE;
        ctx.beginPath();
        for (let i = 0; i < projection.margin_regions.length; i++) {
          const r = projection.margin_regions[i];
          const col = i % G;
          const row = Math.floor(i / G);
          if (col < G - 1) {
            const right = projection.margin_regions[i + 1];
            if (right && Math.sign(right.margin_sign) !== Math.sign(r.margin_sign)) {
              const midWx = (r.wx + right.wx) / 2;
              const { sx, sy: sy1 } = ws(midWx, r.wy + r.cell_h / 2);
              const { sy: sy2 } = ws(midWx, r.wy - r.cell_h / 2);
              ctx.moveTo(sx, sy1);
              ctx.lineTo(sx, sy2);
            }
          }
          if (row < G - 1) {
            const above = projection.margin_regions[i + G];
            if (above && Math.sign(above.margin_sign) !== Math.sign(r.margin_sign)) {
              const midWy = (r.wy + above.wy) / 2;
              const { sx: sx1, sy } = ws(r.wx - r.cell_w / 2, midWy);
              const { sx: sx2 } = ws(r.wx + r.cell_w / 2, midWy);
              ctx.moveTo(sx1, sy);
              ctx.lineTo(sx2, sy);
            }
          }
        }
        ctx.stroke();

        // Margin dashes: edges where margin_sign transitions between +1 and -1 (skipping 0)
        ctx.lineWidth = 1;
        ctx.strokeStyle = T.textSecondary;
        ctx.globalAlpha = 0.55;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < projection.margin_regions.length; i++) {
          const r = projection.margin_regions[i];
          const col = i % G;
          const row = Math.floor(i / G);
          if (col < G - 1) {
            const right = projection.margin_regions[i + 1];
            if (right && r.margin_sign !== 0 && right.margin_sign !== 0 && r.margin_sign !== right.margin_sign) {
              const midWx = (r.wx + right.wx) / 2;
              const { sx, sy: sy1 } = ws(midWx, r.wy + r.cell_h / 2);
              const { sy: sy2 } = ws(midWx, r.wy - r.cell_h / 2);
              ctx.moveTo(sx, sy1);
              ctx.lineTo(sx, sy2);
            }
          }
          if (row < G - 1) {
            const above = projection.margin_regions[i + G];
            if (above && r.margin_sign !== 0 && above.margin_sign !== 0 && r.margin_sign !== above.margin_sign) {
              const midWy = (r.wy + above.wy) / 2;
              const { sx: sx1, sy } = ws(r.wx - r.cell_w / 2, midWy);
              const { sx: sx2 } = ws(r.wx + r.cell_w / 2, midWy);
              ctx.moveTo(sx1, sy);
              ctx.lineTo(sx2, sy);
            }
          }
        }
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
      }
    }

    if (taskType === "regression" && toWY && projection.fit_line?.length) {
      const epsilon = data.summary?.epsilon;

      // Epsilon tube polygon
      if (epsilon != null && toWorldY) {
        const epWY = epsilon / toWorldY.half * 3;
        ctx.beginPath();
        projection.fit_line.forEach((pt, idx) => {
          const { sx, sy } = ws(pt.wx, toWY(pt.wy) + epWY);
          idx === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
        });
        [...projection.fit_line].reverse().forEach((pt) => {
          const { sx, sy } = ws(pt.wx, toWY(pt.wy) - epWY);
          ctx.lineTo(sx, sy);
        });
        ctx.closePath();
        ctx.fillStyle = T.accentFill;
        ctx.fill();
      }

      // Fit line
      ctx.beginPath();
      projection.fit_line.forEach((pt, idx) => {
        const { sx, sy } = ws(pt.wx, toWY(pt.wy));
        idx === 0 ? ctx.moveTo(sx, sy) : ctx.lineTo(sx, sy);
      });
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = T.accent;
      ctx.stroke();
    }

    // Training points
    for (const pt of projection.points) {
      const isSV = pt.is_support_vector;
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      const r = isSV ? 5 : 3.5;

      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }

      ctx.beginPath();
      ctx.arc(sx, sy, r, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = isSV ? 1.75 : 1;
      ctx.strokeStyle = isSV ? T.accent : T.surface;
      ctx.stroke();
    }

    // Query X
    if (projection.query_point) {
      const qWy = taskType === "regression" && toWY && data.prediction?.value != null
        ? toWY(data.prediction.value)
        : projection.query_point.wy;
      drawQueryX(ctx, projection.query_point.wx, qWy, T, ws);
    }
  }, [canvasSize, transform, dark, data]);

  const onMouseMove = (e) => {
    if (!projection || !exploreMode) return;
    if (dragRef.current) {
      const { startX, startY, startOx, startOy, startZoom } = dragRef.current;
      setTransform((prev) => ({
        ...prev,
        ox: startOx + (e.clientX - startX) / startZoom,
        oy: startOy + (e.clientY - startY) / startZoom,
      }));
      setHover(null);
      return;
    }
    const rect = e.currentTarget.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const toWY = toWorldY ? toWorldY.fn : null;
    let best = null, bestDist = 12;
    for (const pt of projection.points) {
      const pwy = taskType === "regression" && toWY ? toWY(pt.label_or_target) : pt.wy;
      const { sx: px, sy: py } = worldToScreen(pt.wx, pwy, zoom, ox, oy, canvasSize.w, canvasSize.h);
      const d = Math.hypot(px - sx, py - sy);
      if (d < bestDist) { bestDist = d; best = { pt, sx, sy }; }
    }
    if (best) {
      setHover({
        sx: best.sx, sy: best.sy,
        item: { lines: [
          { label: "row", value: best.pt.training_index },
          { label: taskType === "regression" ? "target" : "class", value: fmt(best.pt.label_or_target) },
          { label: "support", value: best.pt.is_support_vector ? "yes" : "no" },
        ] },
      });
    } else {
      setHover(null);
    }
  };

  if (!projection) return null;

  const title = taskType === "regression"
    ? "SVR fit · PCA projection"
    : "SVM decision boundary · PCA projection";

  return (
    <div
      style={{ flex: 1, position: "relative", overflow: "hidden" }}
      onMouseMove={onMouseMove}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={() => { dragRef.current = null; setHover(null); }}
      onDoubleClick={() => {
        if (!exploreMode || !projection?.points?.length || !canvasSize.w) return;
        setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", cursor: exploreMode ? "crosshair" : "default" }}
      />
      <div style={{ position: "absolute", top: 8, left: 10, fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", pointerEvents: "none" }}>
        {title}
      </div>
      <button
        onClick={() => setExploreMode((v) => !v)}
        style={{
          position: "absolute",
          bottom: 10,
          left: 10,
          padding: "4px 10px",
          background: exploreMode ? T.accent : T.surface,
          border: `1px solid ${exploreMode ? T.accent : T.border}`,
          borderRadius: 6,
          fontSize: 10,
          fontWeight: 500,
          textTransform: "uppercase",
          letterSpacing: "0.7px",
          color: exploreMode ? "#fff" : T.textTertiary,
          cursor: "pointer",
          fontFamily: FONT_UI,
          transition: "background 150ms ease-out, color 150ms ease-out, border-color 150ms ease-out",
        }}
      >
        {exploreMode ? "exit" : "explore"}
      </button>
      {exploreMode ? (
        <div style={{ position: "absolute", bottom: 10, right: 10, fontSize: 10, color: T.textTertiary, pointerEvents: "none" }}>
          scroll to zoom · drag to pan · double-click to reset · esc to exit
        </div>
      ) : (
        <div style={{ position: "absolute", bottom: 10, left: "50%", transform: "translateX(-50%)", fontSize: 10, color: T.textTertiary, pointerEvents: "none", whiteSpace: "nowrap" }}>
          click explore to interact
        </div>
      )}
      <CanvasTooltip hover={hover} T={T} reduceMotion={reduceMotion} />
    </div>
  );
}
