import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import OverfitCurve from '../../components/visualizations/OverfitCurve'
import { linearTrain, linearOverfit } from '../../api/client'

interface TrainResult {
  train_metrics: { mse: number; mae: number; rmse: number; r2: number }
  test_metrics: { mse: number; mae: number; rmse: number; r2: number }
  X_train: number[]; y_train: number[]; y_pred_train: number[]
  X_test: number[]; y_test: number[]; y_pred_test: number[]
  coefficients: { feature: string; value: number }[]
}

export default function StandardLR() {
  const [noise, setNoise] = useState(0.3)
  const [nSamples, setNSamples] = useState(200)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [overfitData, setOverfitData] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground' | 'overfit'>('theory')

  const train = useCallback(async () => {
    setLoading(true)
    try {
      const [trainRes, overfitRes] = await Promise.all([
        linearTrain({ model_type: 'standard', noise_level: noise, n_samples: nSamples }),
        linearOverfit({ noise_level: noise, n_samples: nSamples }),
      ])
      setResult(trainRes)
      setOverfitData(overfitRes.curve)
    } finally {
      setLoading(false)
    }
  }, [noise, nSamples])

  const tabs = ['theory', 'playground', 'overfit'] as const

  return (
    <div className="animate-slide-up">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Linear Models</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Standard Linear Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">Ordinary Least Squares (OLS) — the foundational regression algorithm that minimizes the sum of squared residuals via the Normal Equation.</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {tabs.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t === 'overfit' ? 'Overfit Sandbox' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* Theory Tab */}
      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">
          <div className="glass-card p-6">
            <h2 className="section-title">How OLS Works</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              Linear Regression models the relationship between a dependent variable <em>y</em> and one or more independent variables <em>X</em> as a linear function. The goal is to find coefficients <strong>β</strong> that minimise the <strong>Residual Sum of Squares (RSS)</strong>.
            </p>
            <FormulaBlock formula="\hat{y} = X\beta = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p" label="Linear Model" />
            <p className="text-sm text-white/60 leading-relaxed mt-4 mb-2">
              We minimise the <strong>cost function</strong> (RSS):
            </p>
            <FormulaBlock formula="J(\beta) = \|y - X\beta\|^2 = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2" label="Cost Function (RSS)" />
            <p className="text-sm text-white/60 leading-relaxed mt-4 mb-2">
              Setting the gradient to zero yields the <strong>Normal Equation</strong> — the closed-form solution:
            </p>
            <FormulaBlock formula="\hat{\beta} = (X^T X)^{-1} X^T y" label="Normal Equation (OLS Solution)" />
            <InfoPanel type="tip" title="When is OLS exact?">
              The Normal Equation computes β directly — no iterative optimization needed. It fails when X<sup>T</sup>X is singular (multicollinear features). Ridge regression fixes this by adding λI to the diagonal.
            </InfoPanel>
          </div>

          <div className="glass-card p-6">
            <h2 className="section-title">Metric Formulas</h2>
            <div className="space-y-2">
              <FormulaBlock formula="\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2" label="Mean Squared Error" />
              <FormulaBlock formula="\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|" label="Mean Absolute Error" />
              <FormulaBlock formula="\text{RMSE} = \sqrt{\text{MSE}}" label="Root Mean Squared Error" />
              <FormulaBlock formula="R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}" label="R-Squared (Coefficient of Determination)" />
            </div>
            <InfoPanel type="info" title="Interpreting R²">
              R² = 1.0 means a perfect fit. R² = 0 means the model performs no better than predicting the mean. R² can be negative if the model is worse than a horizontal line.
            </InfoPanel>
          </div>
        </div>
      )}

      {/* Playground Tab */}
      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Parameters</h2>
              <SliderWithTooltip label="Noise Level" min={0} max={2} step={0.05} value={noise} onChange={setNoise}
                tooltip="Controls the standard deviation of Gaussian noise added to the target variable y. Higher noise → weaker linear relationship → lower R²." />
              <SliderWithTooltip label="Sample Size" min={50} max={500} step={10} value={nSamples} onChange={setNSamples}
                tooltip="Number of training + test samples generated. More data generally improves generalization, reducing the gap between train and test error." />
              <button onClick={train} disabled={loading}
                className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Training…' : '▶ Run Model'}
              </button>
            </div>

            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Test R²" value={result.test_metrics.r2} color="purple" description="Proportion of variance explained" />
                    <MetricCard label="Test RMSE" value={result.test_metrics.rmse} color="blue" description="Root Mean Squared Error" />
                    <MetricCard label="Test MAE" value={result.test_metrics.mae} color="cyan" description="Mean Absolute Error" />
                    <MetricCard label="Test MSE" value={result.test_metrics.mse} color="pink" description="Mean Squared Error" />
                  </div>
                  <ScatterPlot trainX={result.X_train} trainY={result.y_train} testX={result.X_test} testY={result.y_test} trainPred={result.y_pred_train} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">📉</div>
                  <div className="text-white/30 text-sm">Click "Run Model" to generate data and train OLS regression</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Overfit Sandbox */}
      {activeTab === 'overfit' && (
        <div className="animate-fade-in space-y-6">
          <InfoPanel type="warn" title="The Overfitting Experiment">
            Watch how increasing polynomial degree causes the model to memorise training noise (Train MSE → 0) while Test MSE explodes. This is the <strong>bias-variance tradeoff</strong> made visible.
          </InfoPanel>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Inject Chaos</h2>
              <SliderWithTooltip label="Noise Level" min={0} max={2} step={0.05} value={noise} onChange={setNoise}
                tooltip="More noise makes the true signal harder to learn, amplifying the overfitting effect at high polynomial degrees." />
              <SliderWithTooltip label="Sample Size" min={50} max={400} step={10} value={nSamples} onChange={setNSamples}
                tooltip="More samples help the model generalise. Try degree 10 with 50 samples vs 400 — the difference is dramatic." />
              <button onClick={train} disabled={loading}
                className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Generate Curve'}
              </button>
            </div>
            <div className="lg:col-span-2">
              {overfitData.length > 0
                ? <OverfitCurve data={overfitData} xKey="degree" xLabel="Polynomial Degree" isScore={false} />
                : <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                    <div className="text-4xl mb-3">🎢</div>
                    <div className="text-white/30 text-sm">Click "Generate Curve" to see the bias-variance tradeoff</div>
                  </div>
              }
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
