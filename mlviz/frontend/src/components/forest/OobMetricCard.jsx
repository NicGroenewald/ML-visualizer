import Panel from "./Panel";
import { getTheme, FONT_MONO } from "../../theme";

export default function OobMetricCard({ oob, dark }) {
  const T = getTheme(dark);

  if (!oob.available) {
    return (
      <Panel title="out-of-bag" dark={dark}>
        <div style={{ fontSize: 11, color: T.textTertiary, lineHeight: 1.5 }}>
          Out-of-bag metrics unavailable for this model.
        </div>
      </Panel>
    );
  }

  return (
    <Panel title="out-of-bag" dark={dark}>
      <div style={{ display: "flex", gap: 24 }}>
        <div>
          <div style={{ fontSize: 17, fontWeight: 600, color: T.green, fontFamily: FONT_MONO }}>
            {(oob.score * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 10, color: T.textTertiary, marginTop: 2 }}>score</div>
        </div>
        <div>
          <div style={{ fontSize: 17, fontWeight: 600, color: T.red, fontFamily: FONT_MONO }}>
            {(oob.error * 100).toFixed(2)}%
          </div>
          <div style={{ fontSize: 10, color: T.textTertiary, marginTop: 2 }}>error</div>
        </div>
      </div>
    </Panel>
  );
}
