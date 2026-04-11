import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import OverfitCurve from '../../components/visualizations/OverfitCurve'
import { linearTrain, linearOverfit, linearRealDataset } from '../../api/client'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, LineChart, Line, BarChart, Bar, Cell } from 'recharts'

interface TrainResult {
  train_metrics: { mse: number; mae: number; rmse: number; r2: number }
  test_metrics: { mse: number; mae: number; rmse: number; r2: number }
  X_train: number[]; y_train: number[]; y_pred_train: number[]
  X_test: number[]; y_test: number[]; y_pred_test: number[]
  coefficients: { feature: string; value: number }[]
}

const ALL_FEATURES = ['MedIncome', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

export default function StandardLR() {
  const [noise, setNoise] = useState(0.3)
  const [nSamples, setNSamples] = useState(200)
  const [result, setResult] = useState<TrainResult | null>(null)
  const [overfitData, setOverfitData] = useState<any[]>([])
  const [realData, setRealData] = useState<any>(null)
  const [realLoading, setRealLoading] = useState(false)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground' | 'overfit' | 'real-dataset'>('theory')
  const [codeStep, setCodeStep] = useState(0)
  const [testSize, setTestSize] = useState(0.2)
  const [activeFeatures, setActiveFeatures] = useState<string[]>([...ALL_FEATURES])
  const [scoreHistory, setScoreHistory] = useState<{features: string[], trainR2: number, testR2: number, trainRMSE: number, testRMSE: number}[]>([])

  const toggleFeature = (f: string) => {
    setActiveFeatures(prev =>
      prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]
    )
  }

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

  const loadRealDataset = useCallback(async () => {
    if (activeFeatures.length === 0) return
    setRealLoading(true)
    try {
      const res = await linearRealDataset({ test_size: testSize, features: activeFeatures })
      setRealData(res)
      setScoreHistory(prev => [
        ...prev,
        {
          features: [...activeFeatures],
          trainR2: res.train_metrics.r2,
          testR2: res.test_metrics.r2,
          trainRMSE: res.train_metrics.rmse,
          testRMSE: res.test_metrics.rmse,
        }
      ])
    } finally {
      setRealLoading(false)
    }
  }, [testSize, activeFeatures])

  const tabs = ['theory', 'playground', 'overfit', 'real-dataset'] as const

  return (
    <div className="animate-slide-up">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Supervised Learning · Continuous Prediction</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Standard Linear Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">The foundational supervised learning algorithm for predicting continuous values by fitting the best straight line through data.</p>
      </div>

      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {tabs.map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t === 'overfit' ? 'Overfit Sandbox' : t === 'real-dataset' ? '📊 Real Dataset' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {/* ═══════════════════════════ THEORY TAB ═══════════════════════════ */}
      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">

          {/* Introduction */}
          <div className="glass-card p-6">
            <h2 className="section-title">What is Linear Regression?</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              <strong className="text-white/80">Linear Regression</strong> is a fundamental <strong>supervised learning</strong> algorithm used to model the relationship between a <em>dependent variable</em> (target) and one or more <em>independent variables</em> (features). It predicts <strong>continuous values</strong> by fitting a straight line (or hyperplane in multiple dimensions) that best represents the data.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                <div className="text-violet-400 font-semibold text-sm mb-1">📐 Linear Assumption</div>
                <p className="text-[11px] text-white/40 leading-relaxed">It assumes a <strong>linear relationship</strong> between the input features and the output variable.</p>
              </div>
              <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                <div className="text-blue-400 font-semibold text-sm mb-1">📏 Best-Fit Line</div>
                <p className="text-[11px] text-white/40 leading-relaxed">It uses a <strong>best-fit line</strong> (minimizing the distance from each data point to the line) to make predictions.</p>
              </div>
              <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
                <div className="text-cyan-400 font-semibold text-sm mb-1">📈 Use Cases</div>
                <p className="text-[11px] text-white/40 leading-relaxed">Commonly used in <strong>forecasting, trend analysis</strong>, and <strong>predictive modelling</strong> (e.g., predicting house prices, stock prices, temperatures).</p>
              </div>
            </div>
            <InfoPanel type="tip" title="Simple vs Multiple Linear Regression">
              When there is one independent variable, we call it <strong>Simple Linear Regression</strong> (y = β₀ + β₁x). When there are two or more independent variables, it becomes <strong>Multiple Linear Regression</strong> (y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ).
            </InfoPanel>
          </div>

          {/* Visual: How the best-fit line works */}
          <div className="glass-card p-6">
            <h2 className="section-title">How Does the Best-Fit Line Work?</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              The algorithm finds the line that <strong>minimizes the total distance</strong> (error) between every actual data point and the predicted value on the line. These vertical distances are called <strong>residuals</strong>. The model squares them (to penalize large errors more heavily) and sums them up — this is the <strong>Residual Sum of Squares (RSS)</strong>.
            </p>
            <svg viewBox="0 0 400 200" className="w-full h-48 mb-4">
              {/* Grid */}
              <line x1="40" y1="170" x2="380" y2="170" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
              <line x1="40" y1="170" x2="40" y2="20" stroke="rgba(255,255,255,0.1)" strokeWidth="1" />
              
              {/* Regression line */}
              <line x1="60" y1="150" x2="370" y2="40" stroke="#8b5cf6" strokeWidth="2.5" strokeDasharray="0" />
              <text x="375" y="38" fill="#a78bfa" fontSize="10">Best-fit line</text>
              
              {/* Data points and residuals */}
              {[
                [80, 135, 142], [110, 125, 131], [130, 110, 124],
                [160, 130, 113], [190, 95, 103], [210, 110, 96],
                [240, 75, 86], [270, 90, 75], [300, 50, 65],
                [330, 70, 55], [350, 35, 48],
              ].map(([x, actualY, predY], i) => (
                <g key={i}>
                  {/* Residual line */}
                  <line x1={x} y1={actualY} x2={x} y2={predY} stroke="#ef444480" strokeWidth="1" strokeDasharray="3 3" />
                  {/* Actual point */}
                  <circle cx={x} cy={actualY} r="4" fill="#06b6d4" stroke="#06b6d4" strokeWidth="1" opacity="0.8" />
                </g>
              ))}
              
              {/* Labels */}
              <text x="200" y="190" fill="rgba(255,255,255,0.3)" fontSize="10" textAnchor="middle">Feature (X)</text>
              <text x="15" y="100" fill="rgba(255,255,255,0.3)" fontSize="10" textAnchor="middle" transform="rotate(-90, 15, 100)">Target (y)</text>
              
              {/* Legend */}
              <circle cx="80" cy="15" r="4" fill="#06b6d4" />
              <text x="90" y="19" fill="rgba(255,255,255,0.5)" fontSize="9">Actual data points</text>
              <line x1="180" y1="15" x2="200" y2="15" stroke="#ef444480" strokeWidth="1" strokeDasharray="3 3" />
              <text x="205" y="19" fill="rgba(255,255,255,0.5)" fontSize="9">Residuals (errors)</text>
            </svg>
            <p className="text-xs text-white/40 text-center italic">
              The dashed red lines show the residuals — the difference between each actual value and the predicted value on the line. OLS minimizes the sum of the squares of these residuals.
            </p>
          </div>

          {/* Mathematical Foundation */}
          <div className="glass-card p-6">
            <h2 className="section-title">Mathematical Formulation</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              Linear Regression models the relationship between a dependent variable <em>y</em> and one or more independent variables <em>X</em> as a linear function. The goal is to find coefficients <strong>β</strong> that minimize the <strong>Residual Sum of Squares (RSS)</strong>.
            </p>
            <FormulaBlock formula="\hat{y} = X\beta = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p" label="Linear Model" />
            <p className="text-sm text-white/60 leading-relaxed mt-4 mb-2">
              We minimize the <strong>cost function</strong> (RSS):
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

          {/* Key Assumptions */}
          <div className="glass-card p-6">
            <h2 className="section-title">Key Assumptions of Linear Regression</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              For OLS estimates to be <strong>Best Linear Unbiased Estimators (BLUE)</strong>, the following Gauss-Markov assumptions should hold:
            </p>
            <div className="space-y-3">
              {[
                { num: '1', title: 'Linearity', desc: 'The relationship between X and y is linear. Check with a scatter plot before modelling.' },
                { num: '2', title: 'Independence of Errors', desc: 'The residuals (errors) are independent of each other — no autocorrelation.' },
                { num: '3', title: 'Homoscedasticity', desc: 'The variance of residuals is constant across all levels of X. If not (heteroscedasticity), use Weighted Least Squares.' },
                { num: '4', title: 'Normality of Residuals', desc: 'Residuals should be approximately normally distributed. Violations affect confidence intervals, not predictions.' },
                { num: '5', title: 'No Perfect Multicollinearity', desc: 'No feature should be a perfect linear combination of other features. Check with the Variance Inflation Factor (VIF).' },
              ].map(a => (
                <div key={a.num} className="flex gap-3 items-start bg-white/[0.02] rounded-lg p-3 border border-white/[0.04]">
                  <div className="w-7 h-7 rounded-lg bg-violet-600/20 border border-violet-500/30 flex items-center justify-center text-violet-300 text-xs font-bold flex-shrink-0">{a.num}</div>
                  <div>
                    <div className="text-sm font-semibold text-white/80">{a.title}</div>
                    <div className="text-xs text-white/40 leading-relaxed">{a.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Metrics */}
          <div className="glass-card p-6">
            <h2 className="section-title">Evaluation Metrics</h2>
            <p className="text-sm text-white/60 leading-relaxed mb-4">
              How do we know if our line is actually "good"? We measure the error between predictions and actual values using these standardized metrics:
            </p>
            <div className="space-y-2">
              <FormulaBlock formula="\text{MSE} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2" label="Mean Squared Error — penalizes large errors heavily" />
              <FormulaBlock formula="\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i|" label="Mean Absolute Error — intuitive, same units as y" />
              <FormulaBlock formula="\text{RMSE} = \sqrt{\text{MSE}}" label="Root Mean Squared Error — in original units" />
              <FormulaBlock formula="R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}" label="R-Squared (Coefficient of Determination)" />
            </div>
            <InfoPanel type="info" title="Interpreting R²">
              R² = 1.0 means a perfect fit. R² = 0 means the model performs no better than predicting the mean. R² can be negative if the model is worse than a horizontal line. In real datasets, R² of 0.7–0.9 is typically considered a good model.
            </InfoPanel>
          </div>

          {/* When to use / not use */}
          <div className="glass-card p-6">
            <h2 className="section-title">When to Use (and Not Use) Linear Regression</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-emerald-900/10 border border-emerald-500/20 rounded-xl p-4">
                <h4 className="text-emerald-400 font-semibold text-sm mb-2">✅ Good For</h4>
                <ul className="text-xs text-white/50 space-y-1.5">
                  <li>• Predicting house prices based on square footage</li>
                  <li>• Forecasting sales based on advertising spend</li>
                  <li>• Estimating salary from years of experience</li>
                  <li>• Quick baseline model for any regression task</li>
                  <li>• Interpretability — every coefficient has a clear meaning</li>
                </ul>
              </div>
              <div className="bg-red-900/10 border border-red-500/20 rounded-xl p-4">
                <h4 className="text-red-400 font-semibold text-sm mb-2">❌ Not Ideal For</h4>
                <ul className="text-xs text-white/50 space-y-1.5">
                  <li>• Non-linear relationships (use polynomial, trees, or neural nets)</li>
                  <li>• Classification tasks (use Logistic Regression instead)</li>
                  <li>• Data with many outliers (use Robust Regression)</li>
                  <li>• High-dimensional data with multicollinearity (use Ridge/Lasso)</li>
                  <li>• When the relationship changes over time (non-stationary)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ═══════════════════════════ PLAYGROUND TAB ═══════════════════════════ */}
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
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <MetricCard label="Train R²" value={result.train_metrics.r2} color="purple" description="Training fit quality" />
                    <MetricCard label="Test R²" value={result.test_metrics.r2} color="blue" description="Generalization quality" />
                    <MetricCard label="Train RMSE" value={result.train_metrics.rmse} color="cyan" description="Training error" />
                    <MetricCard label="Test RMSE" value={result.test_metrics.rmse} color="pink" description="Test error" />
                  </div>
                  {/* Overfitting/Underfitting indicator */}
                  {(() => {
                    const gap = Math.abs(result.train_metrics.r2 - result.test_metrics.r2)
                    const testR2 = result.test_metrics.r2
                    if (gap > 0.15) return <InfoPanel type="warn" title="⚠️ Possible Overfitting">Train R² is much higher than Test R² (gap: {gap.toFixed(3)}). The model may be memorizing noise in the training data.</InfoPanel>
                    if (testR2 < 0.3) return <InfoPanel type="warn" title="⚠️ Possible Underfitting">Test R² is very low ({testR2.toFixed(3)}). The model may be too simple to capture the pattern, or the data is extremely noisy.</InfoPanel>
                    return <InfoPanel type="tip" title="✅ Good Fit">Train and Test R² are close (gap: {gap.toFixed(3)}), suggesting the model generalizes well.</InfoPanel>
                  })()}
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

      {/* ═══════════════════════════ OVERFIT SANDBOX ═══════════════════════════ */}
      {activeTab === 'overfit' && (
        <div className="animate-fade-in space-y-6">
          <InfoPanel type="warn" title="The Overfitting Experiment">
            Watch how increasing polynomial degree causes the model to memorize training noise (Train MSE → 0) while Test MSE explodes. This is the <strong>bias-variance tradeoff</strong> made visible.
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

      {/* ═══════════════════════════ REAL DATASET TAB ═══════════════════════════ */}
      {activeTab === 'real-dataset' && (
        <div className="animate-fade-in space-y-6">
          <div className="glass-card p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 to-orange-600 flex items-center justify-center text-xl">🏠</div>
              <div>
                <h2 className="text-lg font-bold text-white">California Housing Dataset</h2>
                <p className="text-xs text-white/40">scikit-learn built-in · 20,640 samples · 8 features · Target: Median House Value</p>
              </div>
            </div>
            <p className="text-sm text-white/50 leading-relaxed mb-4">
              This is a <strong>real-world dataset</strong> derived from the 1990 U.S. census. Each row represents a census block group in California. The target variable is the <strong>median house value</strong> (in units of $100,000). You will see the complete end-to-end machine learning workflow below.
            </p>

            {/* Feature Selection */}
            <div className="mb-4 p-4 bg-white/[0.02] border border-white/[0.06] rounded-xl">
              <h4 className="text-sm font-semibold text-white/80 mb-2">🔧 Feature Selection — Toggle features on/off to see impact on scores</h4>
              <p className="text-[11px] text-white/40 mb-3">Click a feature to remove it, then re-run. Watch how R² and RMSE change. Try removing "MedIncome" — the strongest predictor.</p>
              <div className="flex flex-wrap gap-2">
                {ALL_FEATURES.map(f => (
                  <button key={f} onClick={() => toggleFeature(f)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                      activeFeatures.includes(f)
                        ? 'bg-violet-500/20 text-violet-300 border-violet-500/30'
                        : 'bg-red-500/10 text-white/30 border-red-500/20 line-through'
                    }`}>
                    {f}
                  </button>
                ))}
              </div>
              <div className="flex items-center gap-2 mt-2 text-[10px] text-white/30">
                <span className="inline-block w-2 h-2 rounded bg-violet-500/40"></span> Active ({activeFeatures.length})
                <span className="inline-block w-2 h-2 rounded bg-red-500/30 ml-2"></span> Removed ({ALL_FEATURES.length - activeFeatures.length})
              </div>
            </div>

            <div className="flex items-end gap-4">
              <div className="flex-1">
                <SliderWithTooltip label="Test Size" min={0.1} max={0.4} step={0.05} value={testSize} onChange={setTestSize}
                  tooltip="Fraction of data reserved for testing. 0.2 means 80% train / 20% test." />
              </div>
              <button onClick={loadRealDataset} disabled={realLoading || activeFeatures.length === 0} className="btn-primary justify-center px-8 mb-1">
                {realLoading ? '⟳ Training…' : activeFeatures.length === 0 ? '⚠ Select Features' : `▶ Run (${activeFeatures.length} features)`}
              </button>
            </div>
          </div>

          {realData && (
            <>
              {/* Step-by-step Code */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">📝 Step-by-Step Python Code</h3>
                <p className="text-xs text-white/40 mb-4">Follow along — this is the exact code you would write in a Jupyter Notebook to replicate this experiment.</p>
                <div className="flex flex-wrap gap-2 mb-4">
                  {realData.code_steps.map((s: any, i: number) => (
                    <button key={i} onClick={() => setCodeStep(i)}
                      className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${codeStep === i ? 'bg-violet-600 text-white' : 'bg-white/[0.04] text-white/40 hover:text-white/70'}`}>
                      Step {i + 1}
                    </button>
                  ))}
                </div>
                <div className="bg-black/40 rounded-xl p-4 border border-white/[0.06]">
                  <div className="text-xs text-violet-400 font-semibold mb-2">{realData.code_steps[codeStep].title}</div>
                  <pre className="text-xs text-white/70 font-mono whitespace-pre-wrap leading-relaxed overflow-x-auto">{realData.code_steps[codeStep].code}</pre>
                </div>
              </div>

              {/* DataFrame Preview */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">📋 Dataset Preview (First 10 Rows)</h3>
                <p className="text-xs text-white/40 mb-3">Shape: {realData.n_train + realData.n_test} rows × {realData.features.length + 1} columns | Train: {realData.n_train} · Test: {realData.n_test}</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-white/10">
                        {[...realData.features, realData.target].map((col: string) => (
                          <th key={col} className="px-3 py-2 text-left text-white/50 font-medium">{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {realData.preview.map((row: any, i: number) => (
                        <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                          {[...realData.features, realData.target].map((col: string) => (
                            <td key={col} className="px-3 py-2 text-white/60 font-mono">{typeof row[col] === 'number' ? row[col].toFixed(3) : row[col]}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Correlation Matrix */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">🔗 Correlation Matrix</h3>
                <p className="text-xs text-white/40 mb-3">Shows how strongly each feature correlates with every other feature and the target. Values range from -1 (perfect inverse) to +1 (perfect positive).</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-[10px]">
                    <thead>
                      <tr>
                        <th className="px-2 py-1 text-left text-white/50"></th>
                        {realData.correlation.columns.map((col: string) => (
                          <th key={col} className="px-2 py-1 text-white/50 font-medium text-center" style={{writingMode: 'vertical-rl', transform: 'rotate(180deg)', maxHeight: '80px'}}>{col}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {realData.correlation.columns.map((row: string, ri: number) => (
                        <tr key={row}>
                          <td className="px-2 py-1 text-white/50 font-medium">{row}</td>
                          {realData.correlation.values[ri].map((val: number, ci: number) => {
                            const abs = Math.abs(val)
                            const color = val > 0
                              ? `rgba(139, 92, 246, ${abs * 0.6})`
                              : `rgba(239, 68, 68, ${abs * 0.6})`
                            return (
                              <td key={ci} className="px-2 py-1 text-center font-mono text-white/70" style={{backgroundColor: color}}>
                                {val.toFixed(2)}
                              </td>
                            )
                          })}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <InfoPanel type="info" title="Key Insight">
                  Look at the last column (MedHouseVal) — <strong>MedIncome</strong> typically has the highest positive correlation (~0.69) with house value, meaning higher median income strongly predicts higher home prices.
                </InfoPanel>
              </div>

              {/* Train vs Test Metrics */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">📊 Model Performance: Train vs Test</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                  <MetricCard label="Train R²" value={realData.train_metrics.r2} color="purple" description="How well the model fits training data" />
                  <MetricCard label="Test R²" value={realData.test_metrics.r2} color="blue" description="How well the model generalizes" />
                  <MetricCard label="Train RMSE" value={realData.train_metrics.rmse} color="cyan" description="Training prediction error" />
                  <MetricCard label="Test RMSE" value={realData.test_metrics.rmse} color="pink" description="Test prediction error" />
                </div>
                {(() => {
                  const gap = Math.abs(realData.train_metrics.r2 - realData.test_metrics.r2)
                  if (gap > 0.1) return <InfoPanel type="warn" title="⚠️ Overfitting Detected">The Train R² ({realData.train_metrics.r2}) is notably higher than Test R² ({realData.test_metrics.r2}). Consider regularization (Ridge/Lasso) or feature selection.</InfoPanel>
                  if (realData.test_metrics.r2 < 0.4) return <InfoPanel type="warn" title="⚠️ Underfitting">Test R² is low ({realData.test_metrics.r2}). Linear Regression may be too simple for this data. Consider polynomial features or tree-based models.</InfoPanel>
                  return <InfoPanel type="tip" title="✅ Balanced Performance">Train R² ({realData.train_metrics.r2}) ≈ Test R² ({realData.test_metrics.r2}). The model generalizes well with no significant overfitting.</InfoPanel>
                })()}
              </div>

              {/* Coefficients */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">📐 Model Coefficients</h3>
                <p className="text-xs text-white/40 mb-3">Each bar represents how much that feature "pushes" the prediction. Positive = increases house value, Negative = decreases it. Intercept: {realData.intercept}</p>
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={realData.coefficients} layout="vertical" margin={{left: 80, right: 20, top: 10, bottom: 10}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis type="number" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 10}} />
                    <YAxis type="category" dataKey="feature" tick={{fill: 'rgba(255,255,255,0.6)', fontSize: 11}} width={75} />
                    <Tooltip contentStyle={{background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 10}} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {realData.coefficients.map((_: any, i: number) => (
                        <Cell key={i} fill={realData.coefficients[i].value >= 0 ? '#8b5cf6' : '#ef4444'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Actual vs Predicted Scatter */}
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">🎯 Actual vs Predicted (Test Set)</h3>
                <p className="text-xs text-white/40 mb-3">A perfect model would place all points exactly on the diagonal line. The further a point is from the line, the larger the prediction error.</p>
                <ResponsiveContainer width="100%" height={320}>
                  <ScatterChart margin={{top: 10, right: 20, bottom: 30, left: 30}}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="x" name="Actual" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}} label={{value: 'Actual Value', position: 'insideBottom', offset: -15, fill: 'rgba(255,255,255,0.3)', fontSize: 11}} />
                    <YAxis dataKey="y" name="Predicted" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}} label={{value: 'Predicted Value', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.3)', fontSize: 11}} />
                    <Tooltip contentStyle={{background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 10}} />
                    <Scatter
                      data={realData.scatter_actual.map((a: number, i: number) => ({x: a, y: realData.scatter_predicted[i]}))}
                      fill="#8b5cf6" opacity={0.6} r={3}
                    />
                    {/* Perfect prediction line */}
                    <Scatter
                      data={[
                        {x: Math.min(...realData.scatter_actual), y: Math.min(...realData.scatter_actual)},
                        {x: Math.max(...realData.scatter_actual), y: Math.max(...realData.scatter_actual)},
                      ]}
                      fill="none" line={{stroke: '#06b6d4', strokeWidth: 2, strokeDasharray: '6 3'}} r={0}
                    />
                  </ScatterChart>
                </ResponsiveContainer>
                <p className="text-xs text-white/40 text-center italic mt-1">Dashed line = perfect predictions. Points on the line = zero error.</p>
              </div>

              {/* Feature Impact History */}
              {scoreHistory.length > 0 && (
                <div className="glass-card p-6">
                  <h3 className="section-title mb-4">🧪 Feature Impact History</h3>
                  <p className="text-xs text-white/40 mb-3">
                    Each row below is a separate run. Compare how adding/removing features changes the scores. The best Test R² is highlighted in green.
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-white/10">
                          <th className="px-3 py-2 text-left text-white/50 font-medium">#</th>
                          <th className="px-3 py-2 text-left text-white/50 font-medium">Features Used</th>
                          <th className="px-3 py-2 text-center text-white/50 font-medium">Count</th>
                          <th className="px-3 py-2 text-center text-violet-400 font-medium">Train R²</th>
                          <th className="px-3 py-2 text-center text-blue-400 font-medium">Test R²</th>
                          <th className="px-3 py-2 text-center text-cyan-400 font-medium">Train RMSE</th>
                          <th className="px-3 py-2 text-center text-pink-400 font-medium">Test RMSE</th>
                          <th className="px-3 py-2 text-center text-white/50 font-medium">Δ R² (gap)</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(() => {
                          const bestTestR2 = Math.max(...scoreHistory.map(s => s.testR2))
                          return scoreHistory.map((s, i) => {
                            const gap = Math.abs(s.trainR2 - s.testR2)
                            const isBest = s.testR2 === bestTestR2
                            const removedFeatures = ALL_FEATURES.filter(f => !s.features.includes(f))
                            return (
                              <tr key={i} className={`border-b border-white/5 ${isBest ? 'bg-emerald-500/5' : 'hover:bg-white/[0.02]'}`}>
                                <td className="px-3 py-2 text-white/30 font-mono">{i + 1}</td>
                                <td className="px-3 py-2">
                                  <div className="flex flex-wrap gap-1">
                                    {s.features.map(f => (
                                      <span key={f} className="px-1.5 py-0.5 rounded bg-violet-500/10 text-violet-300 text-[9px] font-mono">{f}</span>
                                    ))}
                                    {removedFeatures.map(f => (
                                      <span key={f} className="px-1.5 py-0.5 rounded bg-red-500/10 text-red-400/50 text-[9px] font-mono line-through">{f}</span>
                                    ))}
                                  </div>
                                </td>
                                <td className="px-3 py-2 text-center text-white/40 font-mono">{s.features.length}/{ALL_FEATURES.length}</td>
                                <td className="px-3 py-2 text-center text-violet-300 font-mono">{s.trainR2.toFixed(4)}</td>
                                <td className={`px-3 py-2 text-center font-mono font-semibold ${isBest ? 'text-emerald-400' : 'text-blue-300'}`}>
                                  {s.testR2.toFixed(4)} {isBest && '⭐'}
                                </td>
                                <td className="px-3 py-2 text-center text-cyan-300 font-mono">{s.trainRMSE.toFixed(4)}</td>
                                <td className="px-3 py-2 text-center text-pink-300 font-mono">{s.testRMSE.toFixed(4)}</td>
                                <td className={`px-3 py-2 text-center font-mono ${gap > 0.1 ? 'text-red-400' : gap > 0.05 ? 'text-amber-400' : 'text-emerald-400'}`}>
                                  {gap.toFixed(4)}
                                </td>
                              </tr>
                            )
                          })
                        })()}
                      </tbody>
                    </table>
                  </div>
                  {scoreHistory.length >= 2 && (
                    <div className="mt-4 p-3 rounded-lg bg-white/[0.02] border border-white/[0.06]">
                      <div className="text-xs text-white/50 leading-relaxed">
                        <strong className="text-white/70">💡 Analysis: </strong>
                        {(() => {
                          const first = scoreHistory[0]
                          const last = scoreHistory[scoreHistory.length - 1]
                          const diff = last.testR2 - first.testR2
                          if (diff > 0.01) return `Test R² improved by ${diff.toFixed(4)} from Run 1 → Run ${scoreHistory.length}. Your feature selection helped!`
                          if (diff < -0.01) return `Test R² dropped by ${Math.abs(diff).toFixed(4)} from Run 1 → Run ${scoreHistory.length}. You removed an important feature.`
                          return `Test R² is nearly unchanged (Δ=${diff.toFixed(4)}). The removed features had little predictive value.`
                        })()}
                      </div>
                    </div>
                  )}
                  <button onClick={() => setScoreHistory([])} className="mt-3 text-[10px] text-white/20 hover:text-white/40 transition-colors">
                    Clear history
                  </button>
                </div>
              )}
            </>
          )}

          {!realData && (
            <div className="glass-card p-16 text-center">
              <div className="text-5xl mb-4">🏠</div>
              <div className="text-white/30 text-sm mb-2">Click <strong>"Run Full Pipeline"</strong> above to train on the California Housing dataset</div>
              <div className="text-white/20 text-xs">You'll see the DataFrame, correlation matrix, code walkthrough, and model evaluation — the complete ML workflow.</div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
