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

function ChartFrame({ children, hover, setHover, dark, style = null }) {
  const T = getTheme(dark);
  const reduceMotion = useReducedMotion();
  const ref = useRef(null);
  const transition = reduceMotion ? { duration: 0 } : { duration: 0.15, ease: [0.23, 1, 0.32, 1] };
  return (
    <div
      ref={ref}
      style={{ position: "relative", overflowX: "auto", ...style }}
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

function localPlotScale(points, width, height) {
  const maxAbs = Math.max(
    1,
    ...points.flatMap((p) => [Math.abs(p.wx), Math.abs(p.wy)])
  );
  const pad = 54;
  const usable = Math.min(width, height) - pad * 2;
  const k = usable / (maxAbs * 2);
  return (point) => ({
    sx: width / 2 + point.wx * k,
    sy: height / 2 - point.wy * k,
  });
}

function neighborRows(neighborProjection) {
  const neighbors = neighborProjection?.points?.filter((point) => point.point_type === "neighbor") ?? [];
  const totalWeight = neighbors.reduce((sum, point) => sum + Number(point.effective_weight ?? 0), 0);
  const denominator = totalWeight > 0 ? totalWeight : Math.max(neighbors.length, 1);
  return neighbors.map((point) => {
    const rawWeight = totalWeight > 0 ? Number(point.effective_weight ?? 0) : 1;
    const normalizedWeight = rawWeight / denominator;
    return {
      ...point,
      normalizedWeight,
      contribution: normalizedWeight * Number(point.target_value ?? 0),
    };
  });
}

function contextRows(neighborProjection) {
  return neighborProjection?.points?.filter((point) => point.point_type === "context") ?? [];
}

function targetColor(T, targetRange, value) {
  const [mn, mx] = targetRange ?? [0, 1];
  const t = mx > mn ? (Number(value) - mn) / (mx - mn) : 0.5;
  return mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
}

function pointHoverLines(point, isNeighbor) {
  if (point.point_type === "query") {
    return [
      { label: "role", value: "query" },
      { label: "prediction", value: fmt(point.prediction_value) },
    ];
  }
  if (isNeighbor) {
    return [
      { label: "rank", value: point.rank },
      { label: "row", value: point.training_index },
      { label: "target", value: fmt(point.target_value) },
      { label: "distance", value: fmt(point.distance) },
      { label: "weight", value: fmt(point.effective_weight) },
    ];
  }
  return [
    { label: "row", value: point.training_index },
    { label: "target", value: fmt(point.target_value) },
    { label: "role", value: "context" },
  ];
}

function KnnLocalNeighborPlot({ neighborProjection, dark, activeTrainingIndex, setActiveTrainingIndex }) {
  const T = getTheme(dark);
  const reduceMotion = useReducedMotion();
  const containerRef = useRef(null);
  const { transform, setTransform, dragRef, onMouseDown, onMouseUp } = useCanvasInteraction(containerRef);
  const { zoom, ox, oy } = transform;
  const [hover, setHover] = useState(null);
  const width = 640;
  const height = 360;
  const points = neighborProjection?.points ?? [];
  const query = points.find((point) => point.point_type === "query");
  const neighbors = neighborRows(neighborProjection);
  const context = contextRows(neighborProjection);
  const pointFor = localPlotScale(points, width, height);
  const maxWeight = Math.max(1e-9, ...neighbors.map((point) => point.normalizedWeight));
  const targetRange = extent([
    ...neighbors.map((point) => point.target_value),
    ...context.map((point) => point.target_value),
    neighborProjection?.prediction_value,
  ]);

  if (!query || !neighbors.length) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", color: T.textTertiary, fontSize: 12 }}>
        Local neighbor projection is unavailable for this query.
      </div>
    );
  }

  const queryScreen = pointFor(query);
  const queryHoverPoint = { ...query, prediction_value: neighborProjection?.prediction_value };
  const graphTransform = `translate(${width / 2 + ox} ${height / 2 + oy}) scale(${zoom}) translate(${-width / 2} ${-height / 2})`;
  const screenPointFor = (point) => {
    const base = pointFor(point);
    return {
      sx: width / 2 + ox + zoom * (base.sx - width / 2),
      sy: height / 2 + oy + zoom * (base.sy - height / 2),
    };
  };
  const updateNearestHover = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    const mx = ((event.clientX - rect.left) / rect.width) * width;
    const my = ((event.clientY - rect.top) / rect.height) * height;
    let nearest = null;
    for (const point of [queryHoverPoint, ...neighbors, ...context]) {
      const p = screenPointFor(point);
      const dist = Math.hypot(mx - p.sx, my - p.sy);
      const threshold = point.point_type === "neighbor" ? 15 : point.point_type === "query" ? 16 : 9;
      if (dist <= threshold && (!nearest || dist < nearest.dist)) {
        nearest = { point, dist };
      }
    }
    if (!nearest) {
      setHover(null);
      return;
    }
    if (nearest.point.point_type === "neighbor") {
      setActiveTrainingIndex(nearest.point.training_index);
    }
    setHover({
      sx: event.clientX - rect.left,
      sy: event.clientY - rect.top,
      item: { lines: pointHoverLines(nearest.point, nearest.point.point_type === "neighbor") },
    });
  };
  const movePointer = (event) => {
    if (dragRef.current) {
      const { startX, startY, startOx, startOy, startZoom } = dragRef.current;
      setTransform((prev) => ({
        ...prev,
        ox: startOx + (event.clientX - startX) / startZoom,
        oy: startOy + (event.clientY - startY) / startZoom,
      }));
      return;
    }
    if (!hover) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setHover((current) => current ? {
      ...current,
      sx: event.clientX - rect.left,
      sy: event.clientY - rect.top,
    } : null);
  };

  return (
    <div
      ref={containerRef}
      style={{ width: "100%", height: "100%", position: "relative", overflow: "hidden" }}
      onMouseMove={movePointer}
      onMouseDown={onMouseDown}
      onMouseUp={onMouseUp}
      onMouseLeave={() => { dragRef.current = null; setHover(null); }}
      onDoubleClick={() => { setTransform({ zoom: 1, ox: 0, oy: 0 }); }}
    >
      <svg
        width="100%"
        height="100%"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Query-local KNN neighbor projection"
        onMouseMove={updateNearestHover}
        style={{ cursor: dragRef.current ? "grabbing" : "grab" }}
      >
        <rect x="0" y="0" width={width} height={height} fill={T.canvas} />
        <g transform={graphTransform}>
          {[70, 125, 180].map((radius) => (
            <circle
              key={radius}
              cx={queryScreen.sx}
              cy={queryScreen.sy}
              r={radius}
              fill="none"
              stroke={T.separator}
              strokeWidth={1 / zoom}
            />
          ))}
          {context.map((point) => {
            const p = pointFor(point);
            return (
              <circle
                key={`context-${point.training_index}`}
                cx={p.sx}
                cy={p.sy}
                r={3.2 / Math.sqrt(zoom)}
                fill={targetColor(T, targetRange, point.target_value)}
                opacity="0.34"
                stroke={T.surface}
                strokeWidth={0.75 / zoom}
                pointerEvents="none"
                data-point-type="context"
                data-training-index={point.training_index}
              />
            );
          })}
          {neighbors.map((point) => {
            const p = pointFor(point);
            return (
              <line
                key={`link-${point.training_index}`}
                x1={queryScreen.sx}
                y1={queryScreen.sy}
                x2={p.sx}
                y2={p.sy}
                stroke={T.accent}
                strokeWidth={0.9 / zoom}
                strokeOpacity={0.22 + (point.normalizedWeight / maxWeight) * 0.22}
                strokeLinecap="round"
              />
            );
          })}
          {neighbors.map((point) => {
            const p = pointFor(point);
            const isActive = activeTrainingIndex === point.training_index;
            return (
              <g key={`point-${point.training_index}`}>
                <circle
                  cx={p.sx}
                  cy={p.sy}
                  r={3.2 / Math.sqrt(zoom)}
                  fill={targetColor(T, targetRange, point.target_value)}
                  stroke={isActive ? T.accent : T.surface}
                  strokeWidth={(isActive ? 1.1 : 0.75) / zoom}
                  pointerEvents="none"
                  data-point-type="neighbor"
                  data-training-index={point.training_index}
                  style={{ cursor: "pointer", transition: "stroke 160ms ease-out" }}
                  onFocus={() => setActiveTrainingIndex(point.training_index)}
                  tabIndex="0"
                />
              </g>
            );
          })}
          <g
            pointerEvents="none"
            data-point-type="query"
            style={{ cursor: "crosshair" }}
          >
            <circle cx={queryScreen.sx} cy={queryScreen.sy} r={3.2 / Math.sqrt(zoom)} fill="#F5A524" stroke={T.surface} strokeWidth={1 / zoom} />
            <path
              d={`M ${queryScreen.sx - 3} ${queryScreen.sy - 3} L ${queryScreen.sx + 3} ${queryScreen.sy + 3} M ${queryScreen.sx + 3} ${queryScreen.sy - 3} L ${queryScreen.sx - 3} ${queryScreen.sy + 3}`}
              stroke={T.surface}
              strokeWidth={1 / zoom}
              strokeLinecap="round"
            />
          </g>
        </g>
      </svg>
      <div style={{ position: "absolute", right: 10, bottom: 10, display: "flex", alignItems: "center", gap: 6, background: T.surface, border: `1px solid ${T.border}`, borderRadius: 7, padding: 4 }}>
        {[
          ["−", () => setTransform((value) => { const nz = Math.max(0.55, value.zoom / 1.1); const r = nz / value.zoom; return { zoom: nz, ox: value.ox * r, oy: value.oy * r }; })],
          [`${Math.round(zoom * 100)}%`, null],
          ["+", () => setTransform((value) => { const nz = Math.min(5, value.zoom * 1.1); const r = nz / value.zoom; return { zoom: nz, ox: value.ox * r, oy: value.oy * r }; })],
          ["fit", () => { setTransform({ zoom: 1, ox: 0, oy: 0 }); }],
        ].map(([label, action]) => (
          <button
            key={label}
            type="button"
            onClick={action ?? undefined}
            disabled={!action}
            style={{
              minWidth: label === "fit" ? 28 : 24,
              height: 22,
              border: "none",
              borderLeft: label === "+" || label === "fit" ? `1px solid ${T.separator}` : "none",
              background: "transparent",
              color: action ? T.textSecondary : T.text,
              cursor: action ? "pointer" : "default",
              fontSize: 11,
              fontFamily: FONT_MONO,
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <div style={{ position: "absolute", bottom: 10, left: 12, color: T.textTertiary, fontSize: 10, pointerEvents: "none" }}>
        all training rows are projected with the same true-distance MDS map · ctrl+scroll to zoom · drag to pan
      </div>
      <CanvasTooltip hover={hover} T={T} reduceMotion={reduceMotion} />
    </div>
  );
}

export function KnnWeightedAveragePanel({ data, dark, activeTrainingIndex, setActiveTrainingIndex, showHeading = true }) {
  const T = getTheme(dark);
  const neighborProjection = data.neighbor_projection;
  const rows = neighborRows(neighborProjection);
  const prediction = neighborProjection?.prediction_value ?? data.prediction?.value;

  if (!rows.length || prediction == null) {
    return (
      <div style={{ color: T.textTertiary, fontSize: 12, lineHeight: 1.5 }}>
        Supply a query sample to see the neighbor-average explanation.
      </div>
    );
  }

  const contributionTotal = rows.reduce((sum, row) => sum + row.contribution, 0);
  const active = rows.find((row) => row.training_index === activeTrainingIndex) ?? rows[0];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14, height: "100%" }}>
      <div>
        {showHeading ? (
          <div style={{ fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", marginBottom: 8 }}>
            weighted average
          </div>
        ) : null}
        <div style={{ fontSize: 28, lineHeight: 1, color: T.accent, fontWeight: 650, fontFamily: FONT_MONO }}>
          {fmt(prediction)}
        </div>
        <div style={{ marginTop: 6, color: T.textTertiary, fontSize: 12, lineHeight: 1.45 }}>
          Prediction from the selected neighbors in original feature space.
        </div>
      </div>

      {active && (
        <div style={{ border: `1px solid ${T.border}`, borderRadius: 8, padding: 10, background: T.bg }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, alignItems: "baseline", marginBottom: 9 }}>
            <div style={{ color: T.text, fontSize: 12, fontWeight: 650 }}>
              selected #{active.rank} · row {active.training_index}
            </div>
            <div style={{ color: T.accent, fontSize: 11, fontFamily: FONT_MONO }}>
              {fmt(active.normalizedWeight * 100)}%
            </div>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
            {[
              ["target", active.target_value],
              ["distance", active.distance],
              ["raw weight", active.effective_weight],
              ["contribution", active.contribution],
            ].map(([label, value]) => (
              <div key={label}>
                <div style={{ color: T.text, fontSize: 13, fontFamily: FONT_MONO }}>{fmt(value)}</div>
                <div style={{ color: T.textTertiary, fontSize: 10 }}>{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 8, minHeight: 0, overflowY: "auto" }}>
        {rows.map((row) => (
          <button
            key={row.training_index}
            type="button"
            onMouseEnter={() => setActiveTrainingIndex(row.training_index)}
            onFocus={() => setActiveTrainingIndex(row.training_index)}
            onClick={() => setActiveTrainingIndex(row.training_index)}
            style={{
              border: "none",
              borderTop: `1px solid ${T.separator}`,
              padding: "8px 0 0",
              background: "transparent",
              textAlign: "left",
              cursor: "pointer",
              fontFamily: FONT_UI,
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "baseline" }}>
              <span style={{ color: activeTrainingIndex === row.training_index ? T.accent : T.text, fontSize: 12, fontWeight: 600 }}>#{row.rank} row {row.training_index}</span>
              <span style={{ color: T.textSecondary, fontSize: 11, fontFamily: FONT_MONO }}>{fmt(row.normalizedWeight * 100)}%</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginTop: 6 }}>
              <div>
                <div style={{ color: T.text, fontSize: 12, fontFamily: FONT_MONO }}>{fmt(row.target_value)}</div>
                <div style={{ color: T.textTertiary, fontSize: 10 }}>target</div>
              </div>
              <div>
                <div style={{ color: T.text, fontSize: 12, fontFamily: FONT_MONO }}>{fmt(row.distance)}</div>
                <div style={{ color: T.textTertiary, fontSize: 10 }}>distance</div>
              </div>
              <div>
                <div style={{ color: T.text, fontSize: 12, fontFamily: FONT_MONO }}>{fmt(row.contribution)}</div>
                <div style={{ color: T.textTertiary, fontSize: 10 }}>part</div>
              </div>
            </div>
            <svg width="100%" height="8" viewBox="0 0 100 8" preserveAspectRatio="none" style={{ marginTop: 6, display: "block" }}>
              <rect x="0" y="2" width="100" height="4" rx="2" fill={T.surfaceSecondary} />
              <rect x="0" y="2" width={Math.max(1, row.normalizedWeight * 100)} height="4" rx="2" fill={T.accent} />
            </svg>
          </button>
        ))}
      </div>

      <div style={{ borderTop: `1px solid ${T.separator}`, paddingTop: 10, color: T.textTertiary, fontSize: 11, fontFamily: FONT_MONO }}>
        sum {fmt(contributionTotal)}
      </div>
    </div>
  );
}

export function KnnRegressionExplanation({ data, dark, activeTrainingIndex, setActiveTrainingIndex }) {
  const T = getTheme(dark);

  if (!data.neighbor_projection) {
    return <KnnProjectionPlot data={data} dark={dark} />;
  }

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex", background: T.bg }}>
      <section style={{
        flex: 1,
        minHeight: 320,
        display: "flex",
        minWidth: 0,
      }}>
        <div style={{ flex: 1, minWidth: 0, minHeight: 0, position: "relative", background: T.canvas, width: "100%", height: "100%" }}>
          <div style={{ position: "absolute", top: 10, left: 12, zIndex: 2, fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", pointerEvents: "none" }}>
            local KNN map · true feature-space distances
          </div>
          <KnnLocalNeighborPlot
            neighborProjection={data.neighbor_projection}
            dark={dark}
            activeTrainingIndex={activeTrainingIndex}
            setActiveTrainingIndex={setActiveTrainingIndex}
          />
        </div>
      </section>
    </div>
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

function drawSoftBoundaryGlow(ctx, regions, classColors, boundaryDist, ws, zoom) {
  for (let i = 0; i < regions.length; i++) {
    const r = regions[i];
    const d = boundaryDist[i];
    const alpha = d <= 2 ? 0.22 - (d / 2) * 0.18 : 0.06;
    const { sx: cx, sy: cy } = ws(r.wx, r.wy);
    const pw = r.cell_w * WORLD_SCALE * zoom;
    const ph = r.cell_h * WORLD_SCALE * zoom;
    const [rv, gv, bv] = hexToRgb(classColors[r.class_index] ?? classColors[0]);
    ctx.fillStyle = `rgba(${rv}, ${gv}, ${bv}, ${alpha})`;
    ctx.fillRect(cx - pw / 2, cy - ph / 2, pw, ph);
  }
}

function drawQueryX(ctx, wx, wy, ws) {
  const { sx, sy } = ws(wx, wy);
  const s = 8;
  ctx.lineWidth = 2.5;
  ctx.strokeStyle = "#F5A524";
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(sx - s, sy - s);
  ctx.lineTo(sx + s, sy + s);
  ctx.moveTo(sx + s, sy - s);
  ctx.lineTo(sx - s, sy + s);
  ctx.stroke();
  ctx.lineCap = "butt";
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
            left: (hover.sx ?? hover.x ?? 0) + 12,
            top: Math.max(8, (hover.sy ?? hover.y ?? 60) - 60),
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

function useCanvasInteraction(canvasRef) {
  const [transform, setTransform] = useState({ zoom: 1, ox: 0, oy: 0 });
  const transformRef = useRef(transform);
  const dragRef = useRef(null);

  useEffect(() => { transformRef.current = transform; }, [transform]);

  // Zoom on ctrl+scroll or trackpad pinch (browsers fire ctrlKey=true for pinch).
  // Plain scroll returns early without preventDefault so the page scrolls freely.
  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e) => {
      if (!e.ctrlKey) return;
      e.preventDefault();
      const { zoom, ox, oy } = transformRef.current;
      const factor = e.deltaY < 0 ? 1.02 : 1 / 1.02;
      const newZoom = Math.max(0.2, Math.min(6, zoom * factor));
      const rect = el.getBoundingClientRect();
      const zoomRatio = newZoom / zoom;
      const anchorX = e.clientX - rect.left - rect.width / 2;
      const anchorY = e.clientY - rect.top - rect.height / 2;
      setTransform({
        zoom: newZoom,
        ox: ox * zoomRatio + anchorX * (1 - zoomRatio),
        oy: oy * zoomRatio + anchorY * (1 - zoomRatio),
      });
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [canvasRef]);

  const onMouseDown = (e) => {
    if (e.button !== 0) return;
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
  const { transform, setTransform, dragRef, onMouseDown, onMouseUp } = useCanvasInteraction(canvasRef);
  const { zoom, ox, oy } = transform;
  const [hover, setHover] = useState(null);
  const [showBoundaries, setShowBoundaries] = useState(false);

  const projection = data.projection;
  const taskType = data.task_type ?? "classification";

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

  const boundaryDist = useMemo(() => {
    const regions = projection?.regions;
    if (!regions || taskType !== "classification") return null;
    const G = GRID_SIZE;
    const dist = new Uint8Array(G * G).fill(255);
    const bnd = [];
    for (let i = 0; i < regions.length; i++) {
      const row = Math.floor(i / G), col = i % G;
      let isBnd = false;
      outer: for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          const nr = row + dr, nc = col + dc;
          if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
          if (regions[nr * G + nc].class_index !== regions[i].class_index) { isBnd = true; break outer; }
        }
      }
      if (isBnd) { dist[i] = 0; bnd.push(i); }
    }
    for (const i of bnd) {
      const row = Math.floor(i / G), col = i % G;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = row + dr, nc = col + dc;
        if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
        const ni = nr * G + nc;
        if (dist[ni] > 1) dist[ni] = 1;
      }
    }
    for (let i = 0; i < G * G; i++) {
      if (dist[i] !== 1) continue;
      const row = Math.floor(i / G), col = i % G;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = row + dr, nc = col + dc;
        if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
        const ni = nr * G + nc;
        if (dist[ni] > 2) dist[ni] = 2;
      }
    }
    return dist;
  }, [projection, taskType]);

  useEffect(() => {
    if (!canvasSize.w || !canvasSize.h || !projection) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, canvasSize.w, canvasSize.h);

    const ws = (wx, wy) => worldToScreen(wx, wy, zoom, ox, oy, canvasSize.w, canvasSize.h);
    const ptWy = (p) => (taskType === "regression" && toWorldY ? toWorldY(p.label_or_target) : p.wy);

    ctx.globalAlpha = 1;
    if (showBoundaries && taskType === "classification" && projection.regions && boundaryDist) {
      drawSoftBoundaryGlow(ctx, projection.regions, classColors, boundaryDist, ws, zoom);
    }

    if (projection.query_point && projection.neighbor_indices?.length) {
      const { sx: qsx, sy: qsy } = ws(projection.query_point.wx, projection.query_point.wy);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#FFFFFF";
      ctx.globalAlpha = 0.2;
      ctx.beginPath();
      for (const idx of projection.neighbor_indices) {
        const pt = projection.points[idx];
        if (!pt) continue;
        const { sx, sy } = ws(pt.wx, ptWy(pt));
        ctx.moveTo(qsx, qsy);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
    }

    // Non-neighbors at 0.6 opacity
    ctx.globalAlpha = 0.6;
    for (const pt of projection.points) {
      if (neighborSet.has(pt.training_index)) continue;
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = T.surface;
      ctx.stroke();
    }

    // Neighbors at full opacity with white ring
    ctx.globalAlpha = 1;
    for (const pt of projection.points) {
      if (!neighborSet.has(pt.training_index)) continue;
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();
    }

    if (projection.query_point) {
      const qWy = taskType === "regression" && toWorldY && data.prediction?.value != null
        ? toWorldY(data.prediction.value)
        : projection.query_point.wy;
      drawQueryX(ctx, projection.query_point.wx, qWy, ws);
    }
  }, [canvasSize, transform, dark, data, showBoundaries, boundaryDist]);

  const onMouseMove = (e) => {
    if (!projection) return;
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
        if (!projection?.points?.length || !canvasSize.w) return;
        setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }}
      />
      <div style={{ position: "absolute", top: 8, left: 10, fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", pointerEvents: "none" }}>
        {taskType === "regression" ? "KNN regression overview · PCA projection" : "KNN decision map · PCA projection"}
      </div>
      {taskType === "classification" && (
        <button
          onClick={() => setShowBoundaries((v) => !v)}
          style={{
            position: "absolute", top: 30, left: 8,
            display: "flex", alignItems: "center", gap: 4,
            padding: "3px 8px", borderRadius: 99,
            border: `1px solid ${showBoundaries ? T.accent : T.border}`,
            background: showBoundaries ? T.accentFill : "transparent",
            color: showBoundaries ? T.accent : T.textTertiary,
            fontSize: 10, fontWeight: 500, fontFamily: FONT_UI,
            cursor: "pointer", letterSpacing: "0.5px",
          }}
        >
          <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5">
            <rect x="0.75" y="0.75" width="3.5" height="3.5" rx="0.5" />
            <rect x="5.75" y="0.75" width="3.5" height="3.5" rx="0.5" />
            <rect x="0.75" y="5.75" width="3.5" height="3.5" rx="0.5" />
            <rect x="5.75" y="5.75" width="3.5" height="3.5" rx="0.5" />
          </svg>
          boundaries
        </button>
      )}
      {taskType === "regression" && (
        <div style={{ position: "absolute", bottom: 10, left: 8, fontSize: 10, color: T.textTertiary, pointerEvents: "none" }}>
          positions are PCA projections — neighbour distances are computed in the original feature space
        </div>
      )}
      <div style={{ position: "absolute", bottom: 10, right: 10, fontSize: 10, color: T.textTertiary, pointerEvents: "none" }}>
        ctrl+scroll to zoom · drag to pan · double-click to reset
      </div>
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
  const { transform, setTransform, dragRef, onMouseDown, onMouseUp } = useCanvasInteraction(canvasRef);
  const { zoom, ox, oy } = transform;
  const [hover, setHover] = useState(null);
  const [showBoundaries, setShowBoundaries] = useState(false);

  const projection = data.projection;
  const taskType = data.task_type ?? "classification";

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

  const boundaryDist = useMemo(() => {
    const regions = projection?.regions;
    if (!regions || taskType !== "classification") return null;
    const G = GRID_SIZE;
    const dist = new Uint8Array(G * G).fill(255);
    const bnd = [];
    for (let i = 0; i < regions.length; i++) {
      const row = Math.floor(i / G), col = i % G;
      let isBnd = false;
      outer: for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          if (!dr && !dc) continue;
          const nr = row + dr, nc = col + dc;
          if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
          if (regions[nr * G + nc].class_index !== regions[i].class_index) { isBnd = true; break outer; }
        }
      }
      if (isBnd) { dist[i] = 0; bnd.push(i); }
    }
    for (const i of bnd) {
      const row = Math.floor(i / G), col = i % G;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = row + dr, nc = col + dc;
        if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
        const ni = nr * G + nc;
        if (dist[ni] > 1) dist[ni] = 1;
      }
    }
    for (let i = 0; i < G * G; i++) {
      if (dist[i] !== 1) continue;
      const row = Math.floor(i / G), col = i % G;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        if (!dr && !dc) continue;
        const nr = row + dr, nc = col + dc;
        if (nr < 0 || nr >= G || nc < 0 || nc >= G) continue;
        const ni = nr * G + nc;
        if (dist[ni] > 2) dist[ni] = 2;
      }
    }
    return dist;
  }, [projection, taskType]);

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

    ctx.globalAlpha = 1;

    if (taskType === "classification") {
      // Soft boundary glow
      if (showBoundaries && projection.regions && boundaryDist) {
        drawSoftBoundaryGlow(ctx, projection.regions, classColors, boundaryDist, ws, zoom);
      }

      // Margin boundary and dashed margin lines (binary SVC only)
      if (projection.margin_regions) {
        const G = GRID_SIZE;
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = T.text;
        ctx.globalAlpha = 0.7;
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

        // Margin dashes
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

    // Non-support-vector points at 0.6 opacity
    ctx.globalAlpha = 0.6;
    for (const pt of projection.points) {
      if (pt.is_support_vector) continue;
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 4, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1;
      ctx.strokeStyle = T.surface;
      ctx.stroke();
    }

    // Support vectors at full opacity with white ring
    ctx.globalAlpha = 1;
    for (const pt of projection.points) {
      if (!pt.is_support_vector) continue;
      const { sx, sy } = ws(pt.wx, ptWy(pt));
      let fill;
      if (taskType === "classification") {
        fill = classColors[pt.class_index] ?? T.textSecondary;
      } else {
        const [mn, mx] = targetRange ?? [0, 1];
        const t = mx > mn ? (pt.label_or_target - mn) / (mx - mn) : 0.5;
        fill = mixHex(T.green, T.accent, Math.max(0, Math.min(1, t)));
      }
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#FFFFFF";
      ctx.stroke();
    }

    // Query X
    if (projection.query_point) {
      const qWy = taskType === "regression" && toWY && data.prediction?.value != null
        ? toWY(data.prediction.value)
        : projection.query_point.wy;
      drawQueryX(ctx, projection.query_point.wx, qWy, ws);
    }
  }, [canvasSize, transform, dark, data, showBoundaries, boundaryDist]);

  const onMouseMove = (e) => {
    if (!projection) return;
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
        if (!projection?.points?.length || !canvasSize.w) return;
        setTransform(computeAutoFit(projection.points, canvasSize.w, canvasSize.h));
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ width: "100%", height: "100%", display: "block", cursor: "crosshair" }}
      />
      <div style={{ position: "absolute", top: 8, left: 10, fontSize: 10, color: T.textTertiary, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.7px", pointerEvents: "none" }}>
        {title}
      </div>
      <button
        onClick={() => setShowBoundaries((v) => !v)}
        style={{
          position: "absolute", top: 30, left: 8,
          display: "flex", alignItems: "center", gap: 4,
          padding: "3px 8px", borderRadius: 99,
          border: `1px solid ${showBoundaries ? T.accent : T.border}`,
          background: showBoundaries ? T.accentFill : "transparent",
          color: showBoundaries ? T.accent : T.textTertiary,
          fontSize: 10, fontWeight: 500, fontFamily: FONT_UI,
          cursor: "pointer", letterSpacing: "0.5px",
        }}
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="0.75" y="0.75" width="3.5" height="3.5" rx="0.5" />
          <rect x="5.75" y="0.75" width="3.5" height="3.5" rx="0.5" />
          <rect x="0.75" y="5.75" width="3.5" height="3.5" rx="0.5" />
          <rect x="5.75" y="5.75" width="3.5" height="3.5" rx="0.5" />
        </svg>
        boundaries
      </button>
      <div style={{ position: "absolute", bottom: 10, right: 10, fontSize: 10, color: T.textTertiary, pointerEvents: "none" }}>
        ctrl+scroll to zoom · drag to pan · double-click to reset
      </div>
      <CanvasTooltip hover={hover} T={T} reduceMotion={reduceMotion} />
    </div>
  );
}
