import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import MathTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import { dtTrain, dtOverfitCurve } from '../../api/client'
import OverfitCurve from '../../components/visualizations/OverfitCurve'

export default function DTRegression() {
  const [maxDepth, setMaxDepth] = useState(3)
  const [noise, setNoise] = useState(0.2)
  const [result, setResult] = useState<any>(null)
  const [overfitData, setOverfitData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground'>('theory')

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const [res, ov] = await Promise.all([
        dtTrain({ task: 'regression', max_depth: maxDepth, noise }),
        dtOverfitCurve({ task: 'regression', noise }),
      ])
      setResult(res)
      setOverfitData(ov.curve)
    } finally { setLoading(false) }
  }, [maxDepth, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Tree-Based · Non-parametric</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Decision Trees — Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">Instead of minimizing impurity like classification, regression trees recursively split the feature space to minimize the mean squared error (variance) within each node.</p>
      </div>

      <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {(['theory', 'playground'] as const).map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">
          <div className="glass-card p-6">
            <h2 className="section-title">Splitting Criteria (MSE)</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              A regression tree chooses splits to minimize the variance of the target variable $y$ in the resulting child nodes. The prediction value for a leaf is simply the mean of all training samples in that leaf.
            </p>
            <FormulaBlock formula="\bar{y}_m = \frac{1}{N_m} \sum_{i \in R_m} y_i" label="Prediction at Leaf Node 'm'" />
            <FormulaBlock formula="\text{MSE}_m = \frac{1}{N_m} \sum_{i \in R_m} (y_i - \bar{y}_m)^2" label="Loss Function in Node 'm'" />
            <p className="text-sm text-white/60 mt-4 leading-relaxed">
              The tree splits node $m$ into $left$ and $right$ to minimize:
            </p>
            <FormulaBlock formula="J(feature, threshold) = \frac{N_{left}}{N_m} \text{MSE}_{left} + \frac{N_{right}}{N_m} \text{MSE}_{right}" label="Split Objective" />
          </div>
          <InfoPanel type="warn" title="Extrapolation">
            Unlike linear regression, decision trees can <strong>never</strong> predict values outside the range of their training data. A decision tree trained on $y \in [0, 10]$ will never output 11, making them poor choices for capturing long-term trends in time series.
          </InfoPanel>
        </div>
      )}

      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Parameters</h2>
              <MathTooltip label="Max Depth" min={1} max={15} step={1} value={maxDepth} onChange={setMaxDepth}
                tooltip="Deep trees create highly fragmented 'staircase' predictions trying to hit every point perfectly." />
              <MathTooltip label="Noise" min={0} max={1} step={0.05} value={noise} onChange={setNoise}
                tooltip="More noise means the tree tries to fit randomness if depth is too high." />
              <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Run Regression Tree'}
              </button>
            </div>
            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="R²" value={result.metrics.r2} color="purple" />
                    <MetricCard label="RMSE" value={result.metrics.rmse} color="blue" />
                  </div>
                  <OverfitCurve data={overfitData} xKey="depth" xLabel="Tree Depth" isScore currentX={maxDepth} />
                </>
              ) : (
                <div className="glass-card p-12 flex items-center justify-center text-center">
                  <div className="text-white/30 text-sm">Run model to view performance</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
