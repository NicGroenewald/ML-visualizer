import Panel from "./Panel";
import { getTheme, getClassColors, FONT_MONO } from "../../theme";

export default function VoteDistributionPanel({ voteDistribution, dark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);

  if (voteDistribution === null) {
    return (
      <Panel title="vote distribution" dark={dark}>
        <div style={{ fontSize: 11, color: T.textTertiary, lineHeight: 1.5 }}>
          No query sample supplied — vote distribution is unavailable.
        </div>
      </Panel>
    );
  }

  const winnerColor = classColors[voteDistribution.winning_class_index] ?? T.text;

  return (
    <Panel title="vote distribution" dark={dark}>
      {/* Winner headline */}
      <div style={{
        display: "flex", alignItems: "baseline", gap: 8, marginBottom: 12,
      }}>
        <span style={{ fontSize: 17, fontWeight: 600, color: winnerColor }}>
          {String(voteDistribution.winning_class_label)}
        </span>
        <span style={{ fontSize: 11, color: T.textTertiary, fontFamily: FONT_MONO }}>
          {voteDistribution.total_votes} votes
        </span>
      </div>

      {/* Per-class rows */}
      <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
        {voteDistribution.class_votes.map((c) => (
          <div key={c.class_index}>
            <div style={{
              display: "flex", justifyContent: "space-between",
              alignItems: "baseline", marginBottom: 3, gap: 12,
            }}>
              <span style={{
                fontSize: 11, color: T.textSecondary,
                display: "flex", alignItems: "center", gap: 6,
              }}>
                <span style={{
                  width: 6, height: 6, borderRadius: 2,
                  background: classColors[c.class_index] ?? T.textSecondary,
                  display: "inline-block", flexShrink: 0,
                }} />
                {String(c.class_label)}
              </span>
              <span style={{ fontSize: 11, color: T.text, fontFamily: FONT_MONO, flexShrink: 0 }}>
                {c.votes} · {(c.fraction * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{
              height: 4, borderRadius: 2,
              background: T.surfaceSecondary,
              overflow: "hidden",
            }}>
              <div style={{
                height: "100%", borderRadius: 2,
                width: `${c.fraction * 100}%`,
                background: classColors[c.class_index] ?? T.textSecondary,
                transition: "width 300ms cubic-bezier(0.23, 1, 0.32, 1)",
              }} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
