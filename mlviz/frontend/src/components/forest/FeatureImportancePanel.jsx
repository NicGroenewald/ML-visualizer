import Panel from "./Panel";
import { getTheme, sf, FONT_MONO } from "../../theme";

export default function FeatureImportancePanel({ items, dark }) {
  const T = getTheme(dark);
  const safeItems = Array.isArray(items) ? items : [];

  if (!safeItems.length) {
    return (
      <Panel title="feature importance" dark={dark}>
        <div style={{ fontSize: 11, color: T.textTertiary, lineHeight: 1.5 }}>
          Feature importance unavailable for this model payload.
        </div>
      </Panel>
    );
  }
  const maxImportance = Math.max(...safeItems.map((i) => i.importance), 1e-9);

  return (
    <Panel title="feature importance" dark={dark}>
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {safeItems.map((item) => (
          <div key={item.feature_index}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "baseline", marginBottom: 3, gap: 12,
            }}>
              <span style={{
                fontSize: 11, color: T.textSecondary,
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
              }}>
                {sf(item.feature_name)}
              </span>
              <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO, flexShrink: 0 }}>
                {item.importance.toFixed(4)}
              </span>
            </div>
            <div style={{
              height: 4, borderRadius: 2,
              background: T.surfaceSecondary,
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%", borderRadius: 2,
                width: `${(item.importance / maxImportance) * 100}%`,
                background: T.accent,
                transition: "width 300ms cubic-bezier(0.23, 1, 0.32, 1)",
              }} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
