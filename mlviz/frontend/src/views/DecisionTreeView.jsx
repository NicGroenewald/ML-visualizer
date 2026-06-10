import MLVizTree from "../MLVizTree";
import AppHeader, { Pill, ClassLegend, ThemeToggle } from "../components/AppHeader";
import FeatureImportancePanel from "../components/forest/FeatureImportancePanel";
import { getTheme, getClassColors, FONT_UI } from "../theme";

const SIDEBAR_WIDTH = 296;

export default function DecisionTreeView({ data, dark, toggleDark }) {
  const T = getTheme(dark);
  const classColors = getClassColors(dark);
  const leaves = data.nodes.filter((n) => n.leaf).length;
  const featureImportances = Array.isArray(data.feature_importances)
    ? data.feature_importances
    : [];

  return (
    <div style={{
      background: T.bg,
      display: "flex", flexDirection: "column",
      height: "100vh",
      fontFamily: FONT_UI,
    }}>
      <AppHeader
        dark={dark}
        pills={
          <>
            <Pill T={T} accent>decision tree</Pill>
            {data.dataset_name ? <Pill T={T}>{data.dataset_name}</Pill> : null}
            <Pill T={T}>{data.nodes.length} nodes</Pill>
            <Pill T={T}>{leaves} leaves</Pill>
          </>
        }
        right={
          <>
            <ClassLegend classes={data.classes} classColors={classColors} T={T} />
            <ThemeToggle dark={dark} toggleDark={toggleDark} T={T} />
          </>
        }
      />
      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        <div style={{
          flex: 1, minWidth: 0,
          display: "flex", flexDirection: "column",
        }}>
          <MLVizTree
            nodes={data.nodes}
            path={data.path}
            classes={data.classes}
            dark={dark}
          />
        </div>

        <div style={{
          width: SIDEBAR_WIDTH, flexShrink: 0,
          borderLeft: `1px solid ${T.border}`,
          background: T.bg,
          overflowY: "auto",
          padding: 12,
          display: "flex", flexDirection: "column", gap: 10,
        }}>
          <FeatureImportancePanel items={featureImportances} dark={dark} />
        </div>
      </div>
    </div>
  );
}
