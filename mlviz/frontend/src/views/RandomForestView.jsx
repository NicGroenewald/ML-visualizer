import { useMemo, useState } from "react";
import { motion, useReducedMotion } from "framer-motion";
import MLVizTree from "../MLVizTree";
import ForestHeader from "../components/forest/ForestHeader";
import ForestTreeBrowser from "../components/forest/ForestTreeBrowser";
import FeatureImportancePanel from "../components/forest/FeatureImportancePanel";
import VoteDistributionPanel from "../components/forest/VoteDistributionPanel";
import OobMetricCard from "../components/forest/OobMetricCard";
import { getTheme, FONT_UI } from "../theme";

const SIDEBAR_WIDTH = 296;

export default function RandomForestView({ data, dark, toggleDark }) {
  const T = getTheme(dark);
  const reduceMotion = useReducedMotion();

  const [selectedTreeId, setSelectedTreeId] = useState(
    data.summary.default_tree_id ?? 0
  );
  const selectedTree = useMemo(
    () => data.trees.find((tree) => tree.tree_id === selectedTreeId) ?? data.trees[0],
    [data.trees, selectedTreeId]
  );

  const panelVariants = {
    hidden: { opacity: 0, y: 8 },
    show: (i) => ({
      opacity: 1,
      y: 0,
      transition: reduceMotion
        ? { duration: 0 }
        : { duration: 0.2, delay: i * 0.05, ease: [0.23, 1, 0.32, 1] },
    }),
  };

  return (
    <div style={{
      background: T.bg,
      display: "flex", flexDirection: "column",
      height: "100vh",
      fontFamily: FONT_UI,
    }}>
      <ForestHeader data={data} dark={dark} toggleDark={toggleDark} />

      <ForestTreeBrowser
        trees={data.trees}
        selectedTreeId={selectedTree.tree_id}
        setSelectedTreeId={setSelectedTreeId}
        classes={data.classes}
        dark={dark}
      />

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>
        {/* ── Selected tree canvas ── */}
        <div style={{
          flex: 1, minWidth: 0,
          display: "flex", flexDirection: "column",
        }}>
          <MLVizTree
            key={selectedTree.tree_id}
            nodes={selectedTree.nodes}
            path={selectedTree.path}
            classes={data.classes}
            dark={dark}
          />
        </div>

        {/* ── Ensemble sidebar ── */}
        <div style={{
          width: SIDEBAR_WIDTH, flexShrink: 0,
          borderLeft: `1px solid ${T.border}`,
          background: T.bg,
          overflowY: "auto",
          padding: 12,
          display: "flex", flexDirection: "column", gap: 10,
        }}>
          <motion.div custom={0} initial="hidden" animate="show" variants={panelVariants}>
            <VoteDistributionPanel voteDistribution={data.vote_distribution} dark={dark} />
          </motion.div>
          <motion.div custom={1} initial="hidden" animate="show" variants={panelVariants}>
            <FeatureImportancePanel items={data.feature_importances} dark={dark} />
          </motion.div>
          <motion.div custom={2} initial="hidden" animate="show" variants={panelVariants}>
            <OobMetricCard oob={data.oob} dark={dark} />
          </motion.div>
        </div>
      </div>
    </div>
  );
}
