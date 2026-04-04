import { useState, useCallback, useEffect } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import MathTooltip from '../../components/ui/SliderWithTooltip'
import InfoPanel from '../../components/ui/InfoPanel'
import FeatureImportanceChart from '../../components/visualizations/FeatureImportanceChart'
import CVFoldChart from '../../components/visualizations/CVFoldChart'
import { rfRegression, rfCrossValidate, rfHousingDataset } from '../../api/client'

export default function RFRegression() {
  const [nEstimators, setNEstimators] = useState(50)
  const [maxDepth, setMaxDepth] = useState(10)
  const [datasetPreview, setDatasetPreview] = useState<any>(null)
  const [activeFeatures, setActiveFeatures] = useState<string[]>([])
  const [result, setResult] = useState<any>(null)
  const [cvFolds, setCvFolds] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState<'theory' | 'playground' | 'cv'>('theory')

  useEffect(() => {
    rfHousingDataset().then(data => {
      setDatasetPreview(data)
      setActiveFeatures(data.features)
    })
  }, [])

  const toggleFeature = (f: string) => {
    setActiveFeatures(prev => 
      prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f]
    )
  }

  const run = useCallback(async () => {
    setLoading(true)
    try {
      const res = await rfRegression({ n_estimators: nEstimators, max_depth: maxDepth, active_features: activeFeatures })
      setResult(res)
      if (activeTab === 'cv') {
        const cvRes = await rfCrossValidate({ n_splits: 5, n_estimators: nEstimators, task: 'regression', dataset: 'housing', active_features: activeFeatures })
        setCvFolds(cvRes.folds)
      }
    } finally { setLoading(false) }
  }, [nEstimators, maxDepth, activeFeatures, activeTab])

  // Run CV explicitly when switching to CV tab if not already run
  useEffect(() => {
    if (activeTab === 'cv' && cvFolds.length === 0 && !loading) {
      run()
    }
  }, [activeTab, cvFolds.length, run, loading])

  return (
    <div className="animate-slide-up">
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Regression</span>
          <span className="text-white/30 text-xs">Ensemble · Bagging</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Random Forest Regression</h1>
        <p className="text-white/40 text-sm max-w-2xl">Random Forest trains an ensemble of independent decision trees on random subsets of data and features. The final prediction is the average of all individual trees.</p>
      </div>

      <div className="flex gap-1 p-1 rounded-xl bg-white/[0.04] border border-white/[0.06] mb-8 w-fit">
        {(['theory', 'playground', 'cv'] as const).map(t => (
          <button key={t} onClick={() => setActiveTab(t)}
            className={`px-5 py-2 rounded-lg text-sm font-medium capitalize transition-all ${activeTab === t ? 'bg-violet-600 text-white shadow-glow' : 'text-white/40 hover:text-white/70'}`}>
            {t === 'cv' ? 'Cross Validation' : t.charAt(0).toUpperCase() + t.slice(1)}
          </button>
        ))}
      </div>

      {activeTab === 'theory' && (
        <div className="space-y-6 animate-fade-in">
          <div className="glass-card p-6">
            <h2 className="section-title">Bagging (Bootstrap Aggregation)</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              Decision trees have low bias but high variance. Bagging reduces variance by averaging the predictions of many trees.
            </p>
            <FormulaBlock formula="\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x)" label="Ensemble Prediction" />
            <ul className="list-disc pl-5 text-sm text-white/60 space-y-2 mt-4">
              <li><strong>Bootstrap sampling:</strong> Each tree is trained on a random sample of the training data (drawn with replacement).</li>
              <li><strong>Feature subsampling:</strong> At each split, a random subset of features is considered. This decorrelates the trees so they aren't all making the same splits.</li>
            </ul>
          </div>
          <div className="glass-card p-6">
            <h2 className="section-title">Feature Importance</h2>
            <p className="text-sm text-white/60 mb-4 leading-relaxed">
              Random Forests inherently measure feature importance by tracking how much each feature decreases impurity (e.g., MSE for regression) across all trees. A feature that is consistently chosen for early splits and drastically reduces MSE is considered highly important.
            </p>
          </div>
        </div>
      )}

      {activeTab === 'playground' && (
        <div className="animate-fade-in">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="glass-card p-6">
              <h2 className="section-title mb-6">Parameters</h2>
              <MathTooltip label="Estimators (Trees)" min={10} max={200} step={10} value={nEstimators} onChange={setNEstimators}
                tooltip="Number of trees in the forest. More trees = smoother predictions and less overfitting, but slower to train." />
              <MathTooltip label="Max Depth" min={1} max={30} step={1} value={maxDepth} onChange={setMaxDepth}
                tooltip="Maximum depth of individual trees. Random Forests typically work well with deep trees because the averaging process prevents overfitting." />
              
              <div className="mt-8 mb-4 border-t border-white/10 pt-6">
                <h3 className="text-sm font-semibold text-white/80 mb-2">Feature Selection (California Housing)</h3>
                <p className="text-[11px] text-white/40 mb-3">Drop features to see how the model adapts.</p>
                <div className="flex flex-wrap gap-2">
                  {datasetPreview?.features.map((f: string) => (
                    <button key={f} onClick={() => toggleFeature(f)}
                      className={`px-2 py-1 rounded text-[11px] font-mono transition-colors ${activeFeatures.includes(f) ? 'bg-violet-500/20 text-violet-300 border border-violet-500/30' : 'bg-white/5 text-white/30 border border-white/5 line-through'}`}>
                      {f}
                    </button>
                  ))}
                </div>
              </div>

              <button onClick={run} disabled={loading} className="btn-primary w-full justify-center">
                {loading ? '⟳ Computng…' : '▶ Train Real Data'}
              </button>
            </div>

            <div className="lg:col-span-2 space-y-4">
              {result ? (
                <>
                  <div className="grid grid-cols-2 gap-3">
                    <MetricCard label="Test R²" value={result.metrics.r2} color="purple" description="Generalization performance" />
                    <MetricCard label="Test RMSE" value={result.metrics.rmse} color="blue" />
                  </div>
                  <FeatureImportanceChart data={result.feature_importances} activeFeatures={new Set(activeFeatures)} />
                </>
              ) : (
                <div className="glass-card p-12 flex flex-col items-center justify-center text-center">
                   <div className="text-4xl mb-3">🏔️</div>
                  <div className="text-white/30 text-sm">Train on California Housing dataset to analyze feature importances</div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {activeTab === 'cv' && (
        <div className="animate-fade-in space-y-6">
          <InfoPanel type="info" title="Why Cross-Validation?">
            A single train-test split can be lucky or unlucky. K-Fold Cross-Validation splits the data into K equal chunks (folds). The model is trained K times, each time using K-1 folds for training and 1 fold for validation. 
          </InfoPanel>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
             <div className="glass-card p-6">
                <MathTooltip label="Estimators (Trees)" min={10} max={200} step={10} value={nEstimators} onChange={setNEstimators} tooltip="Number of trees." />
                <button onClick={run} disabled={loading} className="btn-primary w-full mt-4 justify-center">
                  {loading ? '⟳ Running 5 Folds…' : '▶ Run 5-Fold CV'}
                </button>
                <div className="mt-6 text-xs text-white/50 leading-relaxed">
                  <p>In this experiment, we run a 5-fold CV on the California Housing dataset.</p>
                  <p className="mt-2 text-amber-400">Notice the variance between folds. Which fold gives the "true" R²?</p>
                </div>
             </div>
             <div className="lg:col-span-2">
               {cvFolds.length > 0 ? (
                 <CVFoldChart folds={cvFolds} metric="R² Score" />
               ) : (
                 <div className="glass-card p-12 text-center text-white/40">Run 5-Fold CV to see variance.</div>
               )}
             </div>
          </div>
        </div>
      )}
    </div>
  )
}
