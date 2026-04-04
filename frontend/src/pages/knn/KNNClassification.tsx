import { useState, useCallback } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import DecisionBoundary from '../../components/visualizations/DecisionBoundary'
import ConfusionMatrix from '../../components/visualizations/ConfusionMatrix'
import OverfitCurve from '../../components/visualizations/OverfitCurve'
import { knnTrain, knnOverfitCurve } from '../../api/client'

const DATASETS = ['moons', 'circles', 'blobs', 'iris']

export default function KNNClassification() {
  const [k, setK] = useState(5)
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
        knnTrain({ task: 'classification', dataset, n_neighbors: k, noise }),
        knnOverfitCurve({ task: 'classification', dataset, noise }),
      ])
      setResult(res)
      setOverfitData(ov.curve.map((d: any) => ({ ...d, depth: d.k })))
    } finally { setLoading(false) }
  }, [k, dataset, noise])

  const overfitStatus = (() => {
    if (!result) return null
    const acc = result.metrics.accuracy
    if (k <= 2) return { label: '🔴 Overfitting', msg: 'k=1 memorises every training point perfectly but generalises poorly. Decision boundary is jagged.', color: 'text-red-400' }
    if (k >= 20) return { label: '🟡 Underfitting', msg: `k=${k} averages too many neighbours, losing local structure. The boundary becomes too smooth.`, color: 'text-amber-400' }
    return { label: '🟢 Good Fit', msg: `k=${k} balances local sensitivity and smooth generalisation.`, color: 'text-green-400' }
  })()

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-classification">Classification</span>
          <span className="text-white/30 text-xs">Distance-Based · Non-parametric</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">K-Nearest Neighbors — Classification</h1>
        <p className="text-white/40 text-sm max-w-2xl">KNN classifies a point by majority vote among its k nearest training examples. The decision boundary literally "bends around" the data.</p>
      </div>

      <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {(['theory', 'playground', 'overfit'] as const).map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t === 'overfit' ? 'k vs Accuracy' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">
          <div className="glass-card p-6">
            <h2 className="section-title">Distance Metrics</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">KNN's core operation is finding the k points with smallest distance to the query point x_q:</p>
            <FormulaBlock formula="d_{\text{Euclidean}}(x, x') = \sqrt{\sum_{j=1}^p (x_j - x'_j)^2}" label="Euclidean Distance (p=2 Minkowski)" />
            <FormulaBlock formula="d_{\text{Manhattan}}(x, x') = \sum_{j=1}^p |x_j - x'_j|" label="Manhattan Distance (p=1 Minkowski)" />
            <FormulaBlock formula="d_{\text{Minkowski}}(x, x') = \left(\sum_{j=1}^p |x_j - x'_j|^p\right)^{1/p}" label="Minkowski Distance (generalises both)" />
            <InfoPanel type="warn" title="Always scale features before KNN">
              KNN is extremely sensitive to feature scales. A feature ranging 0–10,000 will dominate a feature ranging 0–1 in Euclidean distance. Always normalise/standardise first.
            </InfoPanel>
            <h2 className="section-title mt-6">Prediction Rule</h2>
            <FormulaBlock formula="\hat{y}(x_q) = \underset{c}{\arg\max} \sum_{x_i \in \mathcal{N}_k(x_q)} \mathbf{1}[y_i = c]" label="KNN Classification (Majority Vote)" />
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
              <SliderWithTooltip label="k (Neighbors)" min={1} max={30} step={1} value={k} onChange={setK}
                formatValue={v => String(v)}
                tooltip="Number of nearest neighbors to consider. k=1 → maximum flexibility (memorizes data). Large k → smoother boundary (more bias, less variance)." />
              <SliderWithTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise}
                tooltip="Dataset noise level. Higher noise makes class boundaries overlap." />
              {overfitStatus && (
                <div className={`mt-4 p-3 rounded-xl bg-white/[0.03] border border-white/[0.06] text-xs ${overfitStatus.color}`}>
                  <div className="font-semibold mb-1">{overfitStatus.label}</div>
                  <div className="text-white/40">{overfitStatus.msg}</div>
                </div>
              )}
              <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Run KNN'}
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
                  <ConfusionMatrix matrix={result.metrics.confusion_matrix} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                  <div className="text-4xl mb-3">🎯</div>
                  <div className="text-white/30 text-sm">Run KNN to see Voronoi-style decision boundaries</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'overfit' && (
        <div className="animate-fade-in space-y-6">
          <InfoPanel type="info" title="The k-Overfitting Curve">
            k=1 = zero training error (perfect memorisation). As k increases, the model smooths out, trading variance for bias. The sweet spot is where test accuracy peaks.
          </InfoPanel>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <SliderWithTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise}
                tooltip="More noise widens the gap between k=1 overfitting and optimal k." />
              <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                {loading ? '⟳ Computing…' : '▶ Compute Curve'}
              </button>
            </div>
            <div className="lg:col-span-2">
              {overfitData.length > 0
                ? <OverfitCurve data={overfitData.map(d => ({ ...d, depth: d.k }))} xKey="depth" xLabel="k (Neighbors)" isScore currentX={k} />
                : <div className="glass-card p-12 flex items-center justify-center text-center">
                    <div className="text-white/30 text-sm">Click "Compute Curve" to see accuracy vs k</div>
                  </div>
              }
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
