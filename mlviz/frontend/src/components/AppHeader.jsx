import { getTheme, FONT_UI, FONT_MONO } from "../theme";

export function Pill({ children, T, accent = false }) {
  return (
    <span style={{
      fontSize: 11, padding: "3px 9px", borderRadius: 5,
      background: accent ? T.accentFill : T.surfaceSecondary,
      border: `1px solid ${accent ? T.accent : T.border}`,
      color: accent ? T.accent : T.textSecondary,
      letterSpacing: "0.1px",
      fontFamily: FONT_MONO,
      whiteSpace: "nowrap",
    }}>
      {children}
    </span>
  );
}

export function ClassLegend({ classes, classColors, T }) {
  return (
    <div style={{ display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
      {classes.map((c, idx) => (
        <div key={idx} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11, color: T.textSecondary }}>
          <span style={{
            width: 8, height: 8, borderRadius: 2,
            background: classColors[idx] ?? T.textSecondary,
            display: "inline-block",
          }} />
          {c}
        </div>
      ))}
    </div>
  );
}

export function ThemeToggle({ dark, toggleDark, T }) {
  return (
    <button
      onClick={toggleDark}
      className="zoom-btn"
      style={{
        background: "transparent",
        border: `1px solid ${T.border}`,
        borderRadius: 6,
        padding: "4px 10px",
        cursor: "pointer",
        fontSize: 11,
        color: T.textSecondary,
        letterSpacing: "0.1px",
        whiteSpace: "nowrap",
      }}
      aria-label={dark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {dark ? "☀ light" : "☾ dark"}
    </button>
  );
}

// Shared header shell: wordmark + pills on the left, custom content on the right.
export default function AppHeader({ dark, pills, right }) {
  const T = getTheme(dark);
  return (
    <div style={{
      background: T.surface,
      borderBottom: `1px solid ${T.border}`,
      padding: "10px 16px",
      display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap",
      fontFamily: FONT_UI,
      flexShrink: 0,
    }}>
      <span style={{
        fontSize: 14, fontWeight: 600, color: T.text,
        letterSpacing: "-0.2px", fontFamily: FONT_MONO,
      }}>
        mlviz
      </span>
      <span style={{ width: 1, height: 16, background: T.separator }} />
      {pills}
      <div style={{ marginLeft: "auto", display: "flex", gap: 14, alignItems: "center", flexWrap: "wrap" }}>
        {right}
      </div>
    </div>
  );
}
