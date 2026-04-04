import { useState, useCallback } from 'react'
import MetricCard from '../../components/ui/MetricCard'
import SliderWithTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import FormulaBlock from '../../components/ui/FormulaBlock'
import ScatterPlot from '../../components/visualizations/ScatterPlot'
import { knnTrain } from '../../api/client'

export default function KNNRegression() {
  const [k, setK] = useState(5)
  const [noise, setNoise] = useState(0.3)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const res = await knnTrain({ task: 'regression', n_neighbors: k, noise })
      setResult(res)
    } finally { setLoading(false) }
  }, [k, noise])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Distance-Based · Non-parametric</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">K-Nearest Neighbors — Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">KNN Regression predicts a continuous value as the <em>average</em> of the k nearest training neighbours' target values.</p>
      </div>

      <div className="glass-card p-6 mb-6">
        <h2 className="section-title">Prediction Formula</h2>
        <FormulaBlock formula="\hat{y}(x_q) = \frac{1}{k} \sum_{x_i \in \mathcal{N}_k(x_q)} y_i" label="KNN Regression (Average of k Neighbours)" />
        <InfoPanel type="tip" title="Weighted KNN">
          A common improvement is to weight each neighbour by the inverse of its distance: closer neighbours influence the prediction more. Scikit-learn supports this with weights='distance'.
        </InfoPanel>
        <FormulaBlock formula="\hat{y}(x_q) = \frac{\sum_{i \in \mathcal{N}_k} w_i y_i}{\sum_{i \in \mathcal{N}_k} w_i}, \quad w_i = \frac{1}{d(x_q, x_i)}" label="Distance-Weighted KNN Regression" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="glass-card p-6">
          <h2 className="section-title mb-6">Parameters</h2>
          <SliderWithTooltip label="k (Neighbors)" min={1} max={30} step={1} value={k} onChange={setK}
            formatValue={v => String(v)}
            tooltip="Small k → wiggly predictions (high variance). Large k → flat, over-smoothed predictions (high bias)." />
          <SliderWithTooltip label="Noise" min={0} max={1} step={0.05} value={noise} onChange={setNoise}
            tooltip="Noise injected into the linear target. More noise → lower R²." />
          <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
            {loading ? '⟳ Computing…' : '▶ Run KNN Reg'}
          </button>
        </div>
        <div className="lg:col-span-2 space-y-4">
          {result ? (
            <>
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="R²" value={result.metrics.r2} color="purple" />
                <MetricCard label="RMSE" value={result.metrics.rmse} color="blue" />
                <MetricCard label="MAE" value={result.metrics.mae} color="cyan" />
                <MetricCard label="MSE" value={result.metrics.mse} color="pink" />
              </div>
              <ScatterPlot
                trainX={result.scatter.x_train} trainY={result.scatter.y_train}
                testX={result.scatter.x_test} testY={result.scatter.y_test}
                testPred={result.scatter.y_pred}
              />
            </>
          ) : (
            <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
              <div className="text-4xl mb-3">📐</div>
              <div className="text-white/30 text-sm">Run KNN Regression to see predictions vs actuals</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
