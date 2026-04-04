import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import { linearTrain } from '../../api/client'

export default function RidgeLR() {
  const [alpha, setAlpha] = useState(1.0)
  const [noise, setNoise] = useState(0.3)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground'>('theory')

  const train = useCallback(async () => {
    setLoading(true)
    try {
      const res = await linearTrain({ model_type: 'ridge', alpha, noise_level: noise, n_samples: 200 })
      setResult(res)
    } finally { setLoading(false) }
  }, [alpha, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Regularised · L2</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Ridge Regression (L2)</h1>
        <p className="text-white/40 text-sm max-w-2xl">Ridge adds an L2 penalty — the sum of <em>squared</em> coefficients — to OLS. Unlike Lasso, Ridge never zeros coefficients completely; it shrinks them uniformly, fixing multicollinearity.</p>
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
            <h2 className="section-title">L2 Regularization</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              Ridge penalises large coefficients by adding the <strong>sum of squared coefficients</strong> to the loss:
            </p>
            <FormulaBlock formula="J(\beta) = \|y - X\beta\|^2 + \alpha \sum_{j=1}^{p}\beta_j^2 = \|y - X\beta\|^2 + \alpha\|\beta\|^2_2" label="Ridge Objective" />
            <p className="text-sm text-white/60 mb-4 mt-6 leading-relaxed">
              This has a closed-form solution — simply adding αI to the gram matrix before inversion:
            </p>
            <FormulaBlock formula="\hat{\beta}^{\text{Ridge}} = (X^T X + \alpha I)^{-1} X^T y" label="Ridge Normal Equation" />
            <InfoPanel type="tip" title="Why Ridge fixes multicollinearity">
              When two features are correlated, X<sup>T</sup>X is nearly singular (near-zero eigenvalues), making (X<sup>T</sup>X)<sup>-1</sup> unstable and β explode. Adding αI shifts all eigenvalues by α, making the matrix invertible and coefficients stable.
            </InfoPanel>
            <FormulaBlock formula="\hat{\beta}_j^{\text{Ridge}} = \frac{\hat{\beta}_j^{\text{OLS}}}{1 + \alpha}" label="Shrinkage Factor (Simplified, Orthogonal Case)" />
          </div>
        </div>
      )}

      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Parameters</h2>
              <SliderWithTooltip label="Alpha (λ)" min={0.001} max={10} step={0.001} value={alpha} onChange={setAlpha}
                formatValue={v => v.toFixed(3)}
                tooltip="Regularization strength. α=0 → pure OLS. Large α → all coefficients squeezed toward zero (but never exactly zero, unlike Lasso)." />
              <SliderWithTooltip label="Noise" min={0} max={2} step={0.05} value={noise} onChange={setNoise}
                tooltip="Noise σ in y. Ridge becomes increasingly useful as noise and multicollinearity rise." />
              <button onClick={train} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Training…' : '▶ Run Ridge'}
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
                    <h3 className="section-title mb-3">Coefficients (Ridge shrinks, never zeros)</h3>
                    {result.coefficients.map((c: any) => (
                      <div key={c.feature} className="flex items-center gap-3 mb-2">
                        <span className="text-xs font-mono text-white/50 w-8">{c.feature}</span>
                        <div className="flex-1 h-1.5 rounded-full bg-white/10 relative overflow-hidden">
                          <div className="absolute h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-500"
                            style={{ width: `${Math.min(Math.abs(c.value) / 2 * 100, 100)}%` }} />
                        </div>
                        <span className="text-xs font-mono text-blue-300 w-16 text-right">{c.value.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                  <ScatterPlot trainX={result.X_train} trainY={result.y_train} testX={result.X_test} testY={result.y_test} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">🏔️</div>
                  <div className="text-white/30 text-sm">Run Ridge to see stable coefficient estimates</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
