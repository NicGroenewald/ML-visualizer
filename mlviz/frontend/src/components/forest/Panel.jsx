import { getTheme, FONT_UI } from "../../theme";

// Shared sidebar panel shell for the ensemble views.
export default function Panel({ title, children, dark }) {
  const T = getTheme(dark);
  return (
    <div style={{
      background: T.surface,
      border: `1px solid ${T.border}`,
      borderRadius: 8,
      padding: "13px 14px",
      fontFamily: FONT_UI,
    }}>
      <div style={{
        fontSize: 10, color: T.textTertiary, fontWeight: 500,
        textTransform: "uppercase", letterSpacing: "0.7px",
        marginBottom: 10,
      }}>
        {title}
      </div>
      {children}
    </div>
  );
}
