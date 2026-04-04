import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import MathTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import DecisionBoundary from '../../components/visualizations/DecisionBoundary'
import ConfusionMatrix from '../../components/visualizations/ConfusionMatrix'
import OverfitCurve from '../../components/visualizations/OverfitCurve'
import { dtTrain, dtOverfitCurve } from '../../api/client'

const DATASETS = ['moons', 'circles', 'blobs', 'iris']

export default function DTClassification() {
  const [maxDepth, setMaxDepth] = useState(5)
  const [dataset, setDataset] = useState('moons')
  const [noise, setNoise] = useState(0.2)
  const [result, setResult] = useState<any>(null)
  const [overfitData, setOverfitData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground' | 'overfit'>('theory')

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const [res, ov] = await Promise.all([
        dtTrain({ task: 'classification', dataset, max_depth: maxDepth, noise }),
        dtOverfitCurve({ task: 'classification', dataset, noise }),
      ])
      setResult(res)
      setOverfitData(ov.curve)
    } finally { setLoading(false) }
  }, [maxDepth, dataset, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-classification">Classification</span>
          <span className="text-white/30 text-xs">Tree-Based · Non-parametric</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Decision Trees — Classification</h1>
        <p className="text-white/40 text-sm max-w-2xl">Decision trees recursively split the feature space using orthogonal boundaries to maximize information gain (purity) at each node.</p>
      </div>

      <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {(['theory', 'playground', 'overfit'] as const).map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t === 'overfit' ? 'Depth vs Score' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">
          <div className="glass-card p-6">
            <h2 className="section-title">Splitting Criteria</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              At each step, the tree considers all features and thresholds, choosing the split that minimizes impurity (or maximizes information gain). For classification, common impurity measures are <strong>Gini Impurity</strong> and <strong>Entropy</strong>.
            </p>
            <FormulaBlock formula="Gini(D) = 1 - \sum_{i=1}^C p_i^2" label="Gini Impurity" />
            <FormulaBlock formula="Entropy(D) = - \sum_{i=1}^C p_i \log_2(p_i)" label="Entropy (Information Theory)" />
            <InfoPanel type="tip" title="Gini vs Entropy">
              Gini impurity is slightly faster to compute (no logs) and is the default in scikit-learn. They produce very similar trees in practice. Both are bounded: Gini [0, 0.5] and Entropy [0, 1] for binary classification.
            </InfoPanel>
          </div>
          <div className="glass-card p-6">
            <h2 className="section-title">Information Gain</h2>
            <FormulaBlock formula="IG(D, A) = I(D) - \left( \frac{|D_{left}|}{|D|} I(D_{left}) + \frac{|D_{right}|}{|D|} I(D_{right}) \right)" label="Information Gain" />
            <p className="text-sm text-white/60 mt-4 leading-relaxed">
              Where I(D) is the impurity (Gini or Entropy) of dataset D, and A is the split chosen. The tree stops growing when max_depth is reached, perfectly pure leaves are formed, or min_samples_split conditions are met.
            </p>
          </div>
        </div>
      )}

      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-4">Parameters</h2>
              <div className="mb-4">
                <label className="text-sm font-medium text-white/70 block mb-2">Dataset</label>
                <div className="grid grid-cols-2 gap-2">
                  {DATASETS.map(d => (
                    <button key={d} onClick={() => setDataset(d)}
                      className={`px-3 py-1.5 rounded-lg text-xs capitalize transition-all ${dataset === d ? 'bg-violet-600 text-white' : 'bg-white/[0.04] text-white/40 hover:text-white/70'}`}>
                      {d}
                    </button>
                  ))}
                </div>
              </div>
              <MathTooltip label="Max Depth" min={1} max={15} step={1} value={maxDepth} onChange={setMaxDepth}
                tooltip="Maximum depth of the tree. Unconstrained trees will grow until all leaves are pure (perfect training, terrible test score). Controlling depth is the primary way to prevent overfitting." />
              <MathTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise}
                tooltip="Increases class overlap." />
              <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Build Tree'}
              </button>
            </div>
            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Accuracy" value={result.metrics.accuracy} color="purple" />
                    <MetricCard label="F1 Score" value={result.metrics.f1} color="blue" />
                    <MetricCard label="Precision" value={result.metrics.precision} color="cyan" />
                    <MetricCard label="Recall" value={result.metrics.recall} color="pink" />
                  </div>
                  <DecisionBoundary data={result.boundary} />
                  <InfoPanel type="info" title="Orthogonal Boundaries">
                    Notice how all decision boundaries are strictly vertical or horizontal lines. Decision trees can only split on one feature at a time, creating boxy, staircase-like approximations of curves.
                  </InfoPanel>
                </>
              ) : (
                <div className="glass-card p-12 flex items-center justify-center text-center">
                  <div className="text-white/30 text-sm">Run model to visualize tree splits</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'overfit' && (
        <div className="animate-fade-in space-y-6">
          <InfoPanel type="warn" title="Depth and Overfitting">
            Decision trees are notorious for overfitting. A deep enough tree can perfectly memorize any training set. You will see Train Score stay at 1.0 while Val Score plummets.
          </InfoPanel>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <MathTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise}
                tooltip="More noise means more contradictory points for the tree to memorize, making overfitting worse at high depths." />
              <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Compute Curve'}
              </button>
            </div>
            <div className="lg:col-span-2">
              {overfitData.length > 0
                ? <OverfitCurve data={overfitData} xKey="depth" xLabel="Tree Depth" isScore currentX={maxDepth} />
                : <div className="glass-card p-12 flex items-center justify-center">
                    <div className="text-white/30 text-sm">Click "Compute Curve" to see accuracy vs depth</div>
                  </div>
              }
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
