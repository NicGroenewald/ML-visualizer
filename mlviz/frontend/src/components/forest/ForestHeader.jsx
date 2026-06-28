import AppHeader, { Pill, ClassLegend, ThemeToggle } from "../AppHeader";
import { getTheme, getClassColors, FONT_MONO } from "../../theme";

export default function ForestHeader({ data, dark, toggleDark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const s = data.summary;
  const taskType = data.task_type ?? "classification";
  const isRegression = taskType === "regression";

  return (
    <AppHeader
      dark={dark}
      pills={
        <>
          <Pill T={T} accent>random forest</Pill>
          <Pill T={T}>{taskType}</Pill>
          {data.dataset_name ? <Pill T={T}>{data.dataset_name}</Pill> : null}
          <Pill T={T}>{s.n_estimators} trees</Pill>
          <Pill T={T}>{s.criterion}</Pill>
          <Pill T={T}>max_features: {String(s.max_features)}</Pill>
          {data.oob.available && (
            <span style={{
              fontSize: 11, color: T.textSecondary, fontFamily: FONT_MONO,
              display: "flex", alignItems: "center", gap: 5,
            }}>
              <span style={{
                width: 6, height: 6, borderRadius: "50%",
                background: T.green, display: "inline-block",
              }} />
              {isRegression
                ? `OOB R² ${data.oob.score.toFixed(3)}`
                : `OOB ${(data.oob.score * 100).toFixed(1)}%`}
            </span>
          )}
        </>
      }
      right={
        <>
          {!isRegression && (
            <ClassLegend classes={data.classes} classColors={classColors} T={T} />
          )}
          <ThemeToggle dark={dark} toggleDark={toggleDark} T={T} />
        </>
      }
    />
  );
}
