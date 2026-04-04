import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import { linearTrain } from '../../api/client'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export default function LassoLR() {
  const [alpha, setAlpha] = useState(1.0)
  const [noise, setNoise] = useState(0.3)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground'>('theory')

  const train = useCallback(async () => {
    setLoading(true)
    try {
      const res = await linearTrain({ model_type: 'lasso', alpha, noise_level: noise, n_samples: 200 })
      setResult(res)
    } finally { setLoading(false) }
  }, [alpha, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Regularised · L1</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Lasso Regression (L1)</h1>
        <p className="text-white/40 text-sm max-w-2xl">Lasso adds an L1 penalty to OLS, shrinking coefficients — and driving some to exactly zero, performing automatic feature selection.</p>
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
            <h2 className="section-title">L1 Regularization</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              Lasso (Least Absolute Shrinkage and Selection Operator) modifies the OLS cost function by adding the <strong>sum of absolute values</strong> of coefficients, scaled by λ (alpha):
            </p>
            <FormulaBlock formula="J(\beta) = \underbrace{\|y - X\beta\|^2}_{\text{OLS loss}} + \underbrace{\alpha \sum_{j=1}^{p}|\beta_j|}_{\text{L1 penalty}}" label="Lasso Objective Function" />
            <InfoPanel type="tip" title="Why does L1 create sparsity?">
              The L1 penalty forms a diamond-shaped constraint region in coefficient space. The OLS solution ellipsoid typically touches this diamond at a corner — where one or more coefficients are exactly zero. This is geometrically why Lasso performs feature selection while Ridge (L2, a ball-shaped region) does not.
            </InfoPanel>
            <FormulaBlock formula="\hat{\beta}_j^{\text{Lasso}} = \text{sign}(\hat{\beta}_j^{\text{OLS}}) \cdot \max(|\hat{\beta}_j^{\text{OLS}}| - \alpha, 0)" label="Soft Thresholding (Coordinate Descent Update)" />
            <p className="text-sm text-white/60 leading-relaxed mt-4">
              As α increases: more coefficients become zero → simpler model → higher bias but lower variance. This is the <strong>regularization path</strong>.
            </p>
          </div>
          <div className="glass-card p-6">
            <h2 className="section-title">When to Use Lasso</h2>
            <div className="grid grid-cols-2 gap-4 text-sm text-white/60">
              <InfoPanel type="tip" title="Use Lasso when">You believe only a few features truly matter (sparse ground truth). Great for high-dimensional data (p ≫ n).</InfoPanel>
              <InfoPanel type="warn" title="Avoid Lasso when">Features are correlated — Lasso arbitrarily picks one and zeros the others. Use ElasticNet instead.</InfoPanel>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Parameters</h2>
              <SliderWithTooltip label="Alpha (λ)" min={0.01} max={5} step={0.01} value={alpha} onChange={setAlpha}
                formatValue={v => v.toFixed(2)}
                tooltip="Alpha is the regularization strength (λ). At α=0, Lasso = OLS. As α→∞, all coefficients → 0. Also called the 'L1 penalty weight'." />
              <SliderWithTooltip label="Noise" min={0} max={2} step={0.05} value={noise} onChange={setNoise}
                tooltip="Gaussian noise standard deviation added to y. More noise → harder to learn signal → regularization becomes more important." />
              <button onClick={train} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Training…' : '▶ Run Lasso'}
              </button>
            </div>
            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Test R²" value={result.test_metrics.r2} color="purple" />
                    <MetricCard label="Test RMSE" value={result.test_metrics.rmse} color="blue" />
                    <MetricCard label="Train R²" value={result.train_metrics.r2} color="cyan" />
                    <MetricCard label="Test MAE" value={result.test_metrics.mae} color="pink" />
                  </div>
                  <div className="glass-card p-4">
                    <h3 className="section-title mb-3">Coefficients</h3>
                    <div className="space-y-2">
                      {result.coefficients.map((c: any) => (
                        <div key={c.feature} className="flex items-center gap-3">
                          <span className="text-xs font-mono text-white/50 w-8">{c.feature}</span>
                          <div className="flex-1 h-1.5 rounded-full bg-white/10 relative overflow-hidden">
                            <div className="absolute h-full rounded-full bg-gradient-to-r from-violet-500 to-blue-500"
                              style={{ width: `${Math.min(Math.abs(c.value) / 2 * 100, 100)}%` }} />
                          </div>
                          <span className="text-xs font-mono text-violet-300 w-16 text-right">{c.value.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  <ScatterPlot trainX={result.X_train} trainY={result.y_train} testX={result.X_test} testY={result.y_test} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">🔍</div>
                  <div className="text-white/30 text-sm">Run Lasso to see coefficient shrinkage in action</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
