import { useState, useCallback, useEffect } from 'react'
import FormulaBlock from '../../components/ui/FormulaBlock'
import MetricCard from '../../components/ui/MetricCard'
import InfoPanel from '../../components/ui/InfoPanel'
import { prepDatasetInfo, prepTrain } from '../../api/client'
import { ResponsiveContainer, ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, BarChart, Bar, Cell } from 'recharts'

const NUMERIC_COLS = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
const SCALERS = [
  { id: 'none', label: 'No Scaling', color: 'text-white/40' },
  { id: 'minmax', label: 'Min-Max', color: 'text-emerald-400' },
  { id: 'maxabs', label: 'Max-Abs', color: 'text-blue-400' },
  { id: 'standard', label: 'Standard (Z-score)', color: 'text-violet-400' },
  { id: 'robust', label: 'Robust', color: 'text-amber-400' },
  { id: 'log', label: 'Log Transform', color: 'text-cyan-400' },
]

export default function PreprocessingLR() {
  const [datasetInfo, setDatasetInfo] = useState<any>(null)
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [infoLoading, setInfoLoading] = useState(false)
  const [scalerType, setScalerType] = useState('none')
  const [outlierFeatures, setOutlierFeatures] = useState<string[]>([])
  const [testSize, setTestSize] = useState(0.2)
  const [codeStep, setCodeStep] = useState(0)
  const [uniqueCol, setUniqueCol] = useState('')
  const [scoreHistory, setScoreHistory] = useState<any[]>([])

  useEffect(() => {
    setInfoLoading(true)
    prepDatasetInfo().then(d => { setDatasetInfo(d); setUniqueCol(d.columns[0]) }).finally(() => setInfoLoading(false))
  }, [])

  const toggleOutlier = (f: string) => {
    setOutlierFeatures(prev => prev.includes(f) ? prev.filter(x => x !== f) : [...prev, f])
  }

  const runPipeline = useCallback(async () => {
    setLoading(true)
    try {
      const res = await prepTrain({ scaler_type: scalerType, outlier_features: outlierFeatures, test_size: testSize })
      setResult(res)
      setScoreHistory(prev => [...prev, {
        scaler: scalerType,
        outliers: [...outlierFeatures],
        trainR2: res.train_metrics.r2,
        testR2: res.test_metrics.r2,
        trainRMSE: res.train_metrics.rmse,
        testRMSE: res.test_metrics.rmse,
      }])
    } finally { setLoading(false) }
  }, [scalerType, outlierFeatures, testSize])

  return (
    <div className="animate-slide-up">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <span className="badge badge-regression">Preprocessing</span>
          <span className="text-white/30 text-xs">Linear Regression · Full Pipeline</span>
        </div>
        <h1 className="text-3xl font-bold text-white mb-2">Outlier Handling & Feature Scaling</h1>
        <p className="text-white/40 text-sm max-w-3xl">Learn how to handle outliers, encode categorical data, and scale features before training a Linear Regression model — the complete data preprocessing pipeline using a real Kaggle housing dataset.</p>
      </div>

      {/* ═══════════ SECTION 1: OUTLIER THEORY ═══════════ */}
      <section className="space-y-6 mb-12">
        <div className="glass-card p-6">
          <h2 className="section-title">🔍 What Are Outliers?</h2>
          <p className="text-sm text-white/60 leading-relaxed mb-4">
            <strong className="text-white/80">Outliers</strong> are data points that are significantly different from other observations. They can occur due to measurement errors, data entry mistakes, or genuine extreme values. Outliers can heavily influence the results of a Linear Regression model because OLS minimizes the sum of <strong>squared</strong> residuals — a single extreme point can drag the entire best-fit line toward it.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
            <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
              <div className="text-amber-400 font-semibold text-sm mb-1">⚠️ When to Remove</div>
              <p className="text-[11px] text-white/40 leading-relaxed">When outliers are caused by <strong>errors</strong> (typos, sensor glitches). They add noise and reduce model accuracy.</p>
            </div>
            <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
              <div className="text-emerald-400 font-semibold text-sm mb-1">✅ When to Keep</div>
              <p className="text-[11px] text-white/40 leading-relaxed">When they represent <strong>genuine rare events</strong> (e.g., mansions in a housing dataset). Removing them could bias your model.</p>
            </div>
            <div className="bg-white/[0.03] border border-white/[0.06] rounded-xl p-4">
              <div className="text-violet-400 font-semibold text-sm mb-1">📏 IQR Method</div>
              <p className="text-[11px] text-white/40 leading-relaxed">The <strong>Interquartile Range (IQR)</strong> method is the most common. Points below Q1−1.5×IQR or above Q3+1.5×IQR are flagged.</p>
            </div>
          </div>

          <h3 className="text-sm font-semibold text-white/70 mb-2">IQR Outlier Detection Code:</h3>
          <div className="bg-black/40 rounded-xl p-4 border border-white/[0.06] mb-3">
            <pre className="text-xs text-white/70 font-mono whitespace-pre-wrap leading-relaxed">{`def outlier_detect(x):
    Q1 = x.quantile(0.25)
    Q3 = x.quantile(0.75)
    IQR = Q3 - Q1
    return ((x < Q1 - 1.5 * IQR) | (x > Q3 + 1.5 * IQR))

# Apply to a column
mask = outlier_detect(df['area'])
print(f'Outliers found: {mask.sum()}')
df_clean = df[~mask]  # Remove outliers`}</pre>
          </div>
          <InfoPanel type="tip" title="Train Data Only?">
            Outlier removal should ideally be done <strong>before the train-test split</strong> when outliers represent data quality issues. However, if outliers are genuine rare events, some practitioners prefer removing them only from the training set to keep the test set realistic. In this lab, we remove outliers before splitting.
          </InfoPanel>
        </div>
      </section>

      {/* ═══════════ SECTION 2: FEATURE SCALING THEORY ═══════════ */}
      <section className="space-y-6 mb-12">
        <div className="glass-card p-6">
          <h2 className="section-title">📏 What is Feature Scaling (Normalization)?</h2>
          <p className="text-sm text-white/60 leading-relaxed mb-4">
            <strong className="text-white/80">Feature Scaling</strong> transforms numeric features so they are on a comparable scale. Without scaling, features with large ranges (e.g., area: 1,000–16,000) will dominate over features with small ranges (e.g., bedrooms: 1–6). While Linear Regression coefficients adjust mathematically, <strong>scaling is essential for regularized models (Lasso, Ridge)</strong> and always improves interpretability.
          </p>
          <FormulaBlock formula="\text{Why scale?} \quad x_{\text{area}} \in [1000, 16000] \quad \text{vs} \quad x_{\text{bedrooms}} \in [1, 6]" label="Without scaling, 'area' would dominate the gradient" />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div className="glass-card p-5 border-t-2 border-t-emerald-500">
            <h3 className="text-sm font-bold text-emerald-400 mb-2">1. Min-Max Scaling</h3>
            <FormulaBlock formula="x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}" label="Scales to [0, 1]" />
            <p className="text-[11px] text-white/40 mt-2">Maps values to [0, 1]. Sensitive to outliers. Good for neural networks and image data.</p>
          </div>
          <div className="glass-card p-5 border-t-2 border-t-blue-500">
            <h3 className="text-sm font-bold text-blue-400 mb-2">2. Max-Abs Scaling</h3>
            <FormulaBlock formula="x' = \frac{x}{|x_{\max}|}" label="Scales to [-1, 1]" />
            <p className="text-[11px] text-white/40 mt-2">Divides by the maximum absolute value. Preserves sparsity in sparse datasets. Good for sparse matrices.</p>
          </div>
          <div className="glass-card p-5 border-t-2 border-t-violet-500">
            <h3 className="text-sm font-bold text-violet-400 mb-2">3. Standard (Z-score)</h3>
            <FormulaBlock formula="x' = \frac{x - \mu}{\sigma}" label="Mean=0, Std=1" />
            <p className="text-[11px] text-white/40 mt-2">Centers data at mean 0 with unit variance. Most common. Required for SVM, PCA, and regularized regression.</p>
          </div>
          <div className="glass-card p-5 border-t-2 border-t-amber-500">
            <h3 className="text-sm font-bold text-amber-400 mb-2">4. Robust Scaler</h3>
            <FormulaBlock formula="x' = \frac{x - \text{median}}{Q_3 - Q_1}" label="Uses median & IQR" />
            <p className="text-[11px] text-white/40 mt-2">Uses median and IQR instead of mean/std. <strong>Best choice when outliers are present</strong> — they don't affect the scaling.</p>
          </div>
          <div className="glass-card p-5 border-t-2 border-t-cyan-500">
            <h3 className="text-sm font-bold text-cyan-400 mb-2">5. Log Transform</h3>
            <FormulaBlock formula="x' = \log(1 + x)" label="Compresses large values" />
            <p className="text-[11px] text-white/40 mt-2">Reduces right-skewed distributions. Used for income, population, and price data. log1p handles zeros safely.</p>
          </div>
        </div>

        <InfoPanel type="info" title="Important: Fit on Train, Transform on Test">
          Always <code>fit_transform()</code> on training data and <code>transform()</code> on test data. Never fit the scaler on test data — that causes <strong>data leakage</strong>, making your metrics unrealistically optimistic.
        </InfoPanel>
      </section>

      {/* ═══════════ SECTION 3: CATEGORICAL ENCODING THEORY ═══════════ */}
      <section className="space-y-6 mb-12">
        <div className="glass-card p-6">
          <h2 className="section-title">🏷️ Handling Categorical Features</h2>
          <p className="text-sm text-white/60 leading-relaxed mb-4">
            Machine learning models work with numbers, not text. Categorical features like "mainroad: yes/no" and "furnishingstatus: furnished/semi-furnished/unfurnished" must be converted to numeric values. The encoding method depends on the <strong>type</strong> of categorical data.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="glass-card p-6 border-t-2 border-t-blue-500">
            <h3 className="text-lg font-bold text-blue-400 mb-2">Nominal Data</h3>
            <p className="text-sm text-white/50 mb-3">Categories with <strong>no inherent order</strong>. Examples: mainroad (yes/no), guestroom (yes/no), prefarea (yes/no).</p>
            <p className="text-xs text-white/40 mb-2">Method: <strong>Dummy Variables / One-Hot Encoding</strong> (drop_first=True to avoid multicollinearity)</p>
            <div className="bg-black/40 rounded-xl p-3 border border-white/[0.06]">
              <pre className="text-[11px] text-white/70 font-mono whitespace-pre-wrap leading-relaxed">{`ncf = ['mainroad', 'guestroom', 'basement',
       'hotwaterheating', 'airconditioning', 'prefarea']

X_train = X_train.merge(
    pd.get_dummies(X_train[ncf], dtype=int, drop_first=True),
    left_index=True, right_index=True
)
X_train = X_train.drop(columns=ncf)

X_test = X_test.merge(
    pd.get_dummies(X_test[ncf], dtype=int, drop_first=True),
    left_index=True, right_index=True
)
X_test = X_test.drop(columns=ncf)`}</pre>
            </div>
          </div>

          <div className="glass-card p-6 border-t-2 border-t-amber-500">
            <h3 className="text-lg font-bold text-amber-400 mb-2">Ordinal Data</h3>
            <p className="text-sm text-white/50 mb-3">Categories with a <strong>meaningful order</strong>. Example: furnishingstatus (unfurnished → semi-furnished → furnished).</p>
            <p className="text-xs text-white/40 mb-2">Method: <strong>Ordinal Encoding</strong> — map to integers preserving order.</p>
            <div className="bg-black/40 rounded-xl p-3 border border-white/[0.06]">
              <pre className="text-[11px] text-white/70 font-mono whitespace-pre-wrap leading-relaxed">{`# Map ordinal values to integers
X_train['furnishingstatus'] = X_train['furnishingstatus'].replace(
    {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
)
X_test['furnishingstatus'] = X_test['furnishingstatus'].replace(
    {'unfurnished': 0, 'semi-furnished': 1, 'furnished': 2}
)

# Result:
# unfurnished  → 0 (lowest)
# semi-furnished → 1 (middle)
# furnished    → 2 (highest)`}</pre>
            </div>
          </div>
        </div>

        <InfoPanel type="warn" title="Never use Ordinal Encoding for Nominal Data!">
          If you encode mainroad as yes=1, no=0 — that's fine (binary). But if you encode colors as red=0, blue=1, green=2, the model thinks green {'>'} blue {'>'} red, which is meaningless. Use <strong>One-Hot Encoding</strong> for nominal data with {'>'} 2 categories.
        </InfoPanel>
      </section>

      {/* ═══════════ SECTION 4: INTERACTIVE LAB ═══════════ */}
      <section className="space-y-6">
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 to-red-600 flex items-center justify-center text-xl">🏠</div>
            <div>
              <h2 className="text-lg font-bold text-white">Housing Price Dataset</h2>
              <p className="text-xs text-white/40">545 samples · 13 columns · Target: price · Numeric + Categorical features</p>
            </div>
          </div>

          {/* Dataset Preview */}
          {datasetInfo && (
            <>
              <h3 className="text-sm font-semibold text-white/70 mt-4 mb-2">📋 df.head(10) — DataFrame Preview</h3>
              <div className="overflow-x-auto mb-4">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="border-b border-white/10">
                      {datasetInfo.columns.map((col: string) => (
                        <th key={col} className="px-2 py-1.5 text-left text-white/50 font-medium whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {datasetInfo.preview.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-white/5 hover:bg-white/[0.02]">
                        {datasetInfo.columns.map((col: string) => (
                          <td key={col} className="px-2 py-1.5 text-white/60 font-mono whitespace-nowrap">
                            {typeof row[col] === 'number' ? row[col].toLocaleString() : row[col]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Unique Values Explorer */}
              <h3 className="text-sm font-semibold text-white/70 mb-2">🔎 df[&quot;ColumnName&quot;].unique() — Explore Unique Values</h3>
              <p className="text-[11px] text-white/40 mb-2">Select a column to see its unique values. In Python you would write: <code className="text-violet-300">df[&quot;{uniqueCol}&quot;].unique()</code></p>
              <div className="flex flex-wrap gap-1.5 mb-3">
                {datasetInfo.columns.map((col: string) => (
                  <button key={col} onClick={() => setUniqueCol(col)}
                    className={`px-2.5 py-1 rounded-lg text-[10px] font-mono transition-all border ${
                      uniqueCol === col ? 'bg-violet-600/30 text-violet-300 border-violet-500/40' : 'bg-white/[0.03] text-white/40 border-white/[0.06] hover:text-white/60'
                    }`}>
                    {col}
                  </button>
                ))}
              </div>
              {uniqueCol && datasetInfo.unique_values[uniqueCol] && (
                <div className="bg-black/40 rounded-xl p-3 border border-white/[0.06] mb-4">
                  <div className="text-xs text-violet-400 font-mono mb-1">df[&quot;{uniqueCol}&quot;].unique()  # {datasetInfo.unique_values[uniqueCol].count} unique values</div>
                  <div className="text-xs text-white/60 font-mono">
                    [{datasetInfo.unique_values[uniqueCol].sample.map((v: any) => typeof v === 'string' ? `'${v}'` : v).join(', ')}
                    {datasetInfo.unique_values[uniqueCol].truncated && ', ...'}]
                  </div>
                  <div className="text-[10px] text-white/30 mt-1">Type: {datasetInfo.dtypes[uniqueCol]}</div>
                </div>
              )}
            </>
          )}

          {infoLoading && (
            <div className="text-center py-8 text-white/30 text-sm">⟳ Loading dataset info...</div>
          )}
        </div>

        {/* Controls Panel */}
        <div className="glass-card p-6">
          <h3 className="section-title mb-4">🎛️ Pipeline Controls</h3>

          {/* Outlier Selection */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-white/70 mb-2">Step 1: Outlier Removal (IQR) — Select numeric features</h4>
            <div className="flex flex-wrap gap-2">
              {NUMERIC_COLS.map(f => (
                <button key={f} onClick={() => toggleOutlier(f)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all border ${
                    outlierFeatures.includes(f) ? 'bg-amber-500/20 text-amber-300 border-amber-500/30' : 'bg-white/[0.04] text-white/40 border-white/[0.06]'
                  }`}>
                  {outlierFeatures.includes(f) ? '✓ ' : ''}{f}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-white/30 mt-1">{outlierFeatures.length === 0 ? 'No outlier removal' : `Removing outliers from: ${outlierFeatures.join(', ')}`}</p>
          </div>

          {/* Scaler Selection */}
          <div className="mb-6">
            <h4 className="text-sm font-semibold text-white/70 mb-2">Step 2: Feature Scaling Method</h4>
            <div className="flex flex-wrap gap-2">
              {SCALERS.map(s => (
                <button key={s.id} onClick={() => setScalerType(s.id)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
                    scalerType === s.id ? `bg-violet-600/20 ${s.color} border-violet-500/30` : 'bg-white/[0.04] text-white/40 border-white/[0.06]'
                  }`}>
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="text-xs text-white/40 mb-3">
            <strong>Categorical encoding</strong> is applied automatically: Ordinal encoding for furnishingstatus, OneHotEncoding (drop_first) for yes/no columns.
          </div>

          <button onClick={runPipeline} disabled={loading} className="btn-primary w-full justify-center">
            {loading ? '⟳ Running Pipeline…' : '▶ Run Full Preprocessing Pipeline'}
          </button>
        </div>

        {/* Results */}
        {result && (
          <>
            {/* Outlier Summary */}
            {Object.keys(result.outlier_info).length > 0 && (
              <div className="glass-card p-6">
                <h3 className="section-title mb-3">🔍 Outlier Removal Summary</h3>
                <p className="text-xs text-white/40 mb-3">Rows before: {result.rows_before_outlier} → After: {result.rows_after_outlier} ({result.rows_before_outlier - result.rows_after_outlier} removed)</p>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="px-3 py-2 text-left text-white/50">Feature</th>
                        <th className="px-3 py-2 text-center text-white/50">Q1</th>
                        <th className="px-3 py-2 text-center text-white/50">Q3</th>
                        <th className="px-3 py-2 text-center text-white/50">IQR</th>
                        <th className="px-3 py-2 text-center text-white/50">Lower</th>
                        <th className="px-3 py-2 text-center text-white/50">Upper</th>
                        <th className="px-3 py-2 text-center text-amber-400">Outliers</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.outlier_info).map(([feat, info]: [string, any]) => (
                        <tr key={feat} className="border-b border-white/5">
                          <td className="px-3 py-2 text-white/70 font-mono">{feat}</td>
                          <td className="px-3 py-2 text-center text-white/50 font-mono">{info.Q1}</td>
                          <td className="px-3 py-2 text-center text-white/50 font-mono">{info.Q3}</td>
                          <td className="px-3 py-2 text-center text-white/50 font-mono">{info.IQR}</td>
                          <td className="px-3 py-2 text-center text-white/50 font-mono">{info.lower_bound}</td>
                          <td className="px-3 py-2 text-center text-white/50 font-mono">{info.upper_bound}</td>
                          <td className="px-3 py-2 text-center text-amber-400 font-bold">{info.n_outliers}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Code Walkthrough */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-4">📝 Step-by-Step Python Code</h3>
              <div className="flex flex-wrap gap-2 mb-4">
                {result.code_steps.map((s: any, i: number) => (
                  <button key={i} onClick={() => setCodeStep(i)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${codeStep === i ? 'bg-violet-600 text-white' : 'bg-white/[0.04] text-white/40 hover:text-white/70'}`}>
                    Step {i + 1}
                  </button>
                ))}
              </div>
              <div className="bg-black/40 rounded-xl p-4 border border-white/[0.06]">
                <div className="text-xs text-violet-400 font-semibold mb-2">{result.code_steps[codeStep].title}</div>
                <pre className="text-xs text-white/70 font-mono whitespace-pre-wrap leading-relaxed overflow-x-auto">{result.code_steps[codeStep].code}</pre>
              </div>
            </div>

            {/* DataFrame After Encoding */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-3">📋 DataFrame After Encoding</h3>
              <p className="text-xs text-white/40 mb-2">Features after ordinal + dummy encoding. Columns: {result.feature_names_after_encoding.join(', ')}</p>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="border-b border-white/10">
                      {result.feature_names_after_encoding.map((col: string) => (
                        <th key={col} className="px-2 py-1.5 text-left text-white/50 font-medium whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview_after_encoding.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-white/5">
                        {result.feature_names_after_encoding.map((col: string) => (
                          <td key={col} className="px-2 py-1.5 text-white/60 font-mono whitespace-nowrap">{row[col]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* DataFrame After Scaling */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-3">📏 DataFrame After Scaling ({result.scaler_used})</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-[10px]">
                  <thead>
                    <tr className="border-b border-white/10">
                      {result.feature_names_after_encoding.map((col: string) => (
                        <th key={col} className="px-2 py-1.5 text-left text-white/50 font-medium whitespace-nowrap">{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.preview_after_scaling.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-white/5">
                        {result.feature_names_after_encoding.map((col: string) => (
                          <td key={col} className="px-2 py-1.5 text-white/60 font-mono whitespace-nowrap">{row[col]}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Metrics */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-4">📊 Train vs Test Performance</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
                <MetricCard label="Train R²" value={result.train_metrics.r2} color="purple" description="Training fit" />
                <MetricCard label="Test R²" value={result.test_metrics.r2} color="blue" description="Generalization" />
                <MetricCard label="Train RMSE" value={result.train_metrics.rmse} color="cyan" description="Train error" />
                <MetricCard label="Test RMSE" value={result.test_metrics.rmse} color="pink" description="Test error" />
              </div>
              {(() => {
                const gap = Math.abs(result.train_metrics.r2 - result.test_metrics.r2)
                if (gap > 0.1) return <InfoPanel type="warn" title="⚠️ Overfitting">Train R² ({result.train_metrics.r2}) ≫ Test R² ({result.test_metrics.r2}). Try different scaling or remove outliers.</InfoPanel>
                return <InfoPanel type="tip" title="✅ Balanced">Train R² ({result.train_metrics.r2}) ≈ Test R² ({result.test_metrics.r2}). Good generalization!</InfoPanel>
              })()}
            </div>

            {/* Coefficients */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-4">📐 Coefficients</h3>
              <ResponsiveContainer width="100%" height={Math.max(200, result.coefficients.length * 28)}>
                <BarChart data={result.coefficients} layout="vertical" margin={{left: 100, right: 20, top: 10, bottom: 10}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis type="number" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 10}} />
                  <YAxis type="category" dataKey="feature" tick={{fill: 'rgba(255,255,255,0.6)', fontSize: 10}} width={95} />
                  <Tooltip contentStyle={{background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 10}} />
                  <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                    {result.coefficients.map((_: any, i: number) => (
                      <Cell key={i} fill={result.coefficients[i].value >= 0 ? '#8b5cf6' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Actual vs Predicted */}
            <div className="glass-card p-6">
              <h3 className="section-title mb-4">🎯 Actual vs Predicted (Test)</h3>
              <ResponsiveContainer width="100%" height={320}>
                <ScatterChart margin={{top: 10, right: 20, bottom: 30, left: 30}}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="x" name="Actual" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}} label={{value: 'Actual Price', position: 'insideBottom', offset: -15, fill: 'rgba(255,255,255,0.3)', fontSize: 11}} />
                  <YAxis dataKey="y" name="Predicted" tick={{fill: 'rgba(255,255,255,0.4)', fontSize: 11}} label={{value: 'Predicted Price', angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.3)', fontSize: 11}} />
                  <Tooltip contentStyle={{background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: 10}} />
                  <Scatter data={result.scatter_actual.map((a: number, i: number) => ({x: a, y: result.scatter_predicted[i]}))} fill="#8b5cf6" opacity={0.6} r={3} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>

            {/* Score History */}
            {scoreHistory.length > 0 && (
              <div className="glass-card p-6">
                <h3 className="section-title mb-4">🧪 Preprocessing Impact History</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-white/10">
                        <th className="px-3 py-2 text-left text-white/50">#</th>
                        <th className="px-3 py-2 text-left text-white/50">Scaler</th>
                        <th className="px-3 py-2 text-left text-white/50">Outlier Removal</th>
                        <th className="px-3 py-2 text-center text-violet-400">Train R²</th>
                        <th className="px-3 py-2 text-center text-blue-400">Test R²</th>
                        <th className="px-3 py-2 text-center text-cyan-400">Train RMSE</th>
                        <th className="px-3 py-2 text-center text-pink-400">Test RMSE</th>
                      </tr>
                    </thead>
                    <tbody>
                      {scoreHistory.map((s: any, i: number) => {
                        const best = Math.max(...scoreHistory.map((h: any) => h.testR2))
                        return (
                          <tr key={i} className={`border-b border-white/5 ${s.testR2 === best ? 'bg-emerald-500/5' : ''}`}>
                            <td className="px-3 py-2 text-white/30 font-mono">{i + 1}</td>
                            <td className="px-3 py-2 text-white/60 font-mono">{s.scaler}</td>
                            <td className="px-3 py-2 text-white/50 text-[10px]">{s.outliers.length > 0 ? s.outliers.join(', ') : 'None'}</td>
                            <td className="px-3 py-2 text-center text-violet-300 font-mono">{s.trainR2.toFixed(4)}</td>
                            <td className={`px-3 py-2 text-center font-mono font-semibold ${s.testR2 === best ? 'text-emerald-400' : 'text-blue-300'}`}>
                              {s.testR2.toFixed(4)} {s.testR2 === best && '⭐'}
                            </td>
                            <td className="px-3 py-2 text-center text-cyan-300 font-mono">{s.trainRMSE.toFixed(2)}</td>
                            <td className="px-3 py-2 text-center text-pink-300 font-mono">{s.testRMSE.toFixed(2)}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <button onClick={() => setScoreHistory([])} className="mt-3 text-[10px] text-white/20 hover:text-white/40 transition-colors">Clear history</button>
              </div>
            )}
          </>
        )}

        {!result && !loading && (
          <div className="glass-card p-16 text-center">
            <div className="text-5xl mb-4">🧪</div>
            <div className="text-white/30 text-sm mb-2">Configure outlier removal and scaling above, then click <strong>&quot;Run Full Preprocessing Pipeline&quot;</strong></div>
            <div className="text-white/20 text-xs">You&apos;ll see the full DataFrame, code walkthrough, and how preprocessing impacts model accuracy.</div>
          </div>
        )}
      </section>
    </div>
  )
}
