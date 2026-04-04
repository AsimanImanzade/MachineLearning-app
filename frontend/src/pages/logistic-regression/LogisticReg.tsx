import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import DecisionBoundary from '../../components/visualizations/DecisionBoundary'
import ConfusionMatrix from '../../components/visualizations/ConfusionMatrix'
import { logisticTrain } from '../../api/client'

const DATASETS = ['moons', 'circles', 'blobs', 'iris']

export default function LogisticReg() {
  const [C, setC] = useState(1.0)
  const [noise, setNoise] = useState(0.2)
  const [dataset, setDataset] = useState('moons')
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground'>('theory')

  const train = useCallback(async () => {
    setLoading(true)
    try {
      const res = await logisticTrain({ dataset, C, noise, penalty: 'l2' })
      setResult(res)
    } finally { setLoading(false) }
  }, [C, noise, dataset])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-classification">Classification</span>
          <span className="text-white/30 text-xs">Probabilistic · Sigmoid</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Logistic Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">Despite the name, logistic regression is a <em>classification</em> algorithm. It models the probability of class membership using the sigmoid function.</p>
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
            <h2 className="section-title">The Sigmoid Function</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              Logistic regression squashes a linear combination of features into [0, 1] using the <strong>sigmoid (logistic) function</strong>, interpreting the output as a class probability:
            </p>
            <FormulaBlock formula="\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p" label="Sigmoid Function" />
            <FormulaBlock formula="P(y=1 | X) = \sigma(X\beta) = \frac{1}{1 + e^{-X\beta}}" label="Class Probability" />
            <InfoPanel type="info" title="Log-Odds (Logit)">
              Taking the log of the odds ratio gives the logit: log(p/(1-p)) = Xβ. This is why it's called "logistic" — it models the log-odds as a linear function of X.
            </InfoPanel>
          </div>
          <div className="glass-card p-6">
            <h2 className="section-title">Log-Loss (Cross-Entropy)</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">Unlike MSE, logistic regression maximises the <strong>log-likelihood</strong>, or equivalently minimises <strong>Binary Cross-Entropy Loss</strong>:</p>
            <FormulaBlock formula="\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[ y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i) \right]" label="Binary Cross-Entropy Loss" />
            <FormulaBlock formula="\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}, \quad F_1 = \frac{2 \cdot P \cdot R}{P + R}" label="Classification Metrics" />
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
              <SliderWithTooltip label="C (Inverse Regularization)" min={0.01} max={20} step={0.01} value={C} onChange={setC}
                formatValue={v => v.toFixed(2)}
                tooltip="C = 1/λ (inverse of regularization strength). Small C → strong regularization → simpler boundary. Large C → weak regularization → can overfit complex boundaries." />
              <SliderWithTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise}
                tooltip="Noise in the dataset generation. Higher noise makes classes overlap more, reducing achievable accuracy." />
              <button onClick={train} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Training…' : '▶ Classify'}
              </button>
            </div>

            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Accuracy" value={result.metrics.accuracy} color="purple" description="Correct predictions / Total" />
                    <MetricCard label="F1 Score" value={result.metrics.f1} color="blue" description="Harmonic mean of Precision & Recall" />
                    <MetricCard label="Precision" value={result.metrics.precision} color="cyan" description="TP / (TP + FP)" />
                    <MetricCard label="Recall" value={result.metrics.recall} color="pink" description="TP / (TP + FN)" />
                  </div>
                  <DecisionBoundary data={result.boundary} />
                  <ConfusionMatrix matrix={result.metrics.confusion_matrix} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">🔀</div>
                  <div className="text-white/30 text-sm">Select a dataset and click Classify to see the decision boundary</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
