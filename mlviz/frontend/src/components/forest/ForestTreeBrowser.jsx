import { getTheme, getClassColors, FONT_UI, FONT_MONO } from "../../theme";

// Horizontal strip of compact per-tree cards. Selecting a card swaps the
// tree shown in the main canvas.
export default function ForestTreeBrowser({
  trees,
  selectedTreeId,
  setSelectedTreeId,
  classes,
  dark,
}) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);

  return (
    <div style={{
      display: "flex", gap: 8,
      overflowX: "auto",
      padding: "10px 16px",
      background: T.surface,
      borderBottom: `1px solid ${T.border}`,
      flexShrink: 0,
      fontFamily: FONT_UI,
    }}>
      {trees.map((tree) => {
        const selected = tree.tree_id === selectedTreeId;
        const vote = tree.prediction;
        return (
          <button
            key={tree.tree_id}
            onClick={() => setSelectedTreeId(tree.tree_id)}
            style={{
              background: selected ? T.accentFill : T.surfaceSecondary,
              border: `1px solid ${selected ? T.accent : T.border}`,
              borderRadius: 7,
              padding: "7px 12px",
              cursor: "pointer",
              display: "flex", flexDirection: "column", gap: 4,
              alignItems: "flex-start",
              minWidth: 108,
              flexShrink: 0,
              textAlign: "left",
            }}
            aria-pressed={selected}
            aria-label={`Select tree ${tree.tree_id}`}
          >
            <span style={{
              fontSize: 11, fontWeight: 600,
              color: selected ? T.accent : T.text,
              fontFamily: FONT_MONO,
            }}>
              tree {tree.tree_id}
            </span>
            <span style={{ fontSize: 10, color: T.textTertiary, fontFamily: FONT_MONO }}>
              d{tree.max_depth} · {tree.n_leaves} leaves
            </span>
            {vote && (
              <span style={{
                fontSize: 10, display: "flex", alignItems: "center", gap: 4,
                color: classColors[vote.class_index] ?? T.textSecondary,
                fontWeight: 500,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: 2,
                  background: classColors[vote.class_index] ?? T.textSecondary,
                  display: "inline-block", flexShrink: 0,
                }} />
                {String(vote.class_label)}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
