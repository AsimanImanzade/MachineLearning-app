import { useState, useCallback } from 'react'
import MetricCard from '../../components/ui/MetricCard'
import MathTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import DecisionBoundary from '../../components/visualizations/DecisionBoundary'
import ConfusionMatrix from '../../components/visualizations/ConfusionMatrix'
import { rfClassification } from '../../api/client'

const DATASETS = ['moons', 'circles', 'blobs', 'iris']

export default function RFClassification() {
  const [nEstimators, setNEstimators] = useState(50)
  const [maxDepth, setMaxDepth] = useState(10)
  const [dataset, setDataset] = useState('moons')
  const [noise, setNoise] = useState(0.2)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const res = await rfClassification({ dataset, n_estimators: nEstimators, max_depth: maxDepth, noise })
      setResult(res)
    } finally { setLoading(false) }
  }, [nEstimators, maxDepth, dataset, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-classification">Classification</span>
          <span className="text-white/30 text-xs">Ensemble · Bagging</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Random Forest Classification</h1>
        <p className="text-white/40 text-sm max-w-2xl">Bagging applied to decision tree classifiers. The final class prediction is determined by a majority vote among all the trees.</p>
      </div>

      <InfoPanel type="tip" title="Smoother Boundaries">
        Single decision trees have rigid, orthogonal decision boundaries. Look at how Random Forest creates much smoother, more complex boundaries by averaging the "staircase" splits of hundreds of independent trees.
      </InfoPanel>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
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
          <MathTooltip label="Estimators (Trees)" min={1} max={100} step={1} value={nEstimators} onChange={setNEstimators}
            tooltip="Try setting this to 1: you get a raw decision tree. Watch the boundary smooth out as you push this to 100." />
          <MathTooltip label="Max Depth" min={1} max={20} step={1} value={maxDepth} onChange={setMaxDepth}
             tooltip="Limiting depth per tree still helps prevent total overfitting, though bagging is quite robust." />
          <MathTooltip label="Noise" min={0} max={0.5} step={0.01} value={noise} onChange={setNoise} tooltip="Data overlap." />
          <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
            {loading ? '⟳ Computing…' : '▶ Train Ensemble'}
          </button>
        </div>

        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Accuracy" value={result.metrics.accuracy} color="purple" />
                <MetricCard label="F1 Score" value={result.metrics.f1} color="blue" />
              </div>
              <DecisionBoundary data={result.boundary} />
              <ConfusionMatrix matrix={result.metrics.confusion_matrix} />
            </>
          ) : (
            <div className="glass-card p-12 text-center text-white/40">Run model to view classification results</div>
          )}
        </div>
      </div>
    </div>
  )
}
