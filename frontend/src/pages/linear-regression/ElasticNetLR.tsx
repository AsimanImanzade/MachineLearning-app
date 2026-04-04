import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import { linearTrain } from '../../api/client'

export default function ElasticNetLR() {
  const [alpha, setAlpha] = useState(1.0)
  const [l1Ratio, setL1Ratio] = useState(0.5)
  const [noise, setNoise] = useState(0.3)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground'>('theory')

  const train = useCallback(async () => {
    setLoading(true)
    try {
      const res = await linearTrain({ model_type: 'elasticnet', alpha, l1_ratio: l1Ratio, noise_level: noise, n_samples: 200 })
      setResult(res)
    } finally { setLoading(false) }
  }, [alpha, l1Ratio, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">L1 + L2 Combined</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">ElasticNet Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">ElasticNet linearly combines L1 (Lasso) and L2 (Ridge) penalties, controlled by the <code className="font-mono text-violet-300">l1_ratio</code> parameter. It gets the best of both: sparsity + stability.</p>
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
            <h2 className="section-title">Combined L1 + L2 Penalty</h2>
            <FormulaBlock formula="J(\beta) = \|y - X\beta\|^2 + \alpha \left[ \rho \sum_j|\beta_j| + \frac{1-\rho}{2}\sum_j\beta_j^2 \right]" label="ElasticNet Objective (ρ = l1_ratio)" />
            <div className="grid grid-cols-3 gap-3 mt-6">
              <div className="glass-card p-3 text-center border border-white/[0.06]">
                <div className="text-xs text-white/30 mb-1">ρ = 0</div>
                <div className="text-sm font-medium text-blue-300">Pure Ridge</div>
                <div className="text-[11px] text-white/30 mt-1">No sparsity</div>
              </div>
              <div className="glass-card p-3 text-center border border-violet-500/30 bg-violet-500/5">
                <div className="text-xs text-violet-300 mb-1">0 &lt; ρ &lt; 1</div>
                <div className="text-sm font-medium text-violet-300">ElasticNet</div>
                <div className="text-[11px] text-white/30 mt-1">Best of both</div>
              </div>
              <div className="glass-card p-3 text-center border border-white/[0.06]">
                <div className="text-xs text-white/30 mb-1">ρ = 1</div>
                <div className="text-sm font-medium text-cyan-300">Pure Lasso</div>
                <div className="text-[11px] text-white/30 mt-1">Maximum sparsity</div>
              </div>
            </div>
            <InfoPanel type="tip" title="When to use ElasticNet">
              When you have correlated features (Ridge's strength) but also want feature selection (Lasso's strength). Common in genomics (thousands of correlated genes) and NLP.
            </InfoPanel>
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
                tooltip="Overall regularisation strength. Both L1 and L2 penalties scale with alpha." />
              <SliderWithTooltip label="L1 Ratio (ρ)" min={0} max={1} step={0.01} value={l1Ratio} onChange={setL1Ratio}
                formatValue={v => v.toFixed(2)}
                tooltip="Mixing parameter: 0=Ridge only, 1=Lasso only, 0.5=equal mix. As ρ increases, more coefficients are driven to exactly zero." color="#06b6d4" />
              <SliderWithTooltip label="Noise" min={0} max={2} step={0.05} value={noise} onChange={setNoise}
                tooltip="Gaussian noise σ added to y." />
              <button onClick={train} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Training…' : '▶ Run ElasticNet'}
              </button>
              <div className="mt-4 p-3 bg-violet-500/5 rounded-xl border border-violet-500/20 text-xs text-white/40">
                <div className="font-medium text-violet-300 mb-1">Current Mix</div>
                <div>L1 weight: <span className="text-violet-300">{(l1Ratio * 100).toFixed(0)}%</span></div>
                <div>L2 weight: <span className="text-blue-300">{((1 - l1Ratio) * 100).toFixed(0)}%</span></div>
              </div>
            </div>
            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Test R²" value={result.test_metrics.r2} color="purple" />
                    <MetricCard label="Test RMSE" value={result.test_metrics.rmse} color="blue" />
                    <MetricCard label="Train R²" value={result.train_metrics.r2} color="cyan" />
                    <MetricCard label="Zero Coefficients" value={result.coefficients.filter((c: any) => Math.abs(c.value) < 1e-4).length} color="pink" />
                  </div>
                  <ScatterPlot trainX={result.X_train} trainY={result.y_train} testX={result.X_test} testY={result.y_test} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">⚡</div>
                  <div className="text-white/30 text-sm">Adjust the L1 ratio slider to blend Lasso and Ridge</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
