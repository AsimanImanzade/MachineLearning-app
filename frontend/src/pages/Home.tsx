import { NavLink } from 'react-router-dom'
import SupervisedDiagram from '../components/visualizations/SupervisedDiagram'
import MLFlowchart from '../components/visualizations/MLFlowchart'

const MODULES = [
  { title: 'Linear Regression', desc: 'OLS, Lasso, Ridge, ElasticNet — from the normal equation to regularization paths.', icon: '📈', path: '/linear/standard', color: 'from-violet-600 to-blue-600', tag: 'REG' },
  { title: 'Logistic Regression', desc: 'Sigmoid, log-loss, decision boundaries — probabilistic binary classification.', icon: '🔀', path: '/logistic', color: 'from-blue-600 to-cyan-600', tag: 'CLF' },
  { title: 'K-Nearest Neighbors', desc: 'Distance metrics, the k-overfitting curve, and Voronoi decision boundaries.', icon: '🎯', path: '/knn/classification', color: 'from-cyan-600 to-teal-600', tag: 'REG/CLF' },
  { title: 'Decision Trees', desc: 'Gini impurity, information gain, and why max_depth is your best friend.', icon: '🌳', path: '/decision-tree/classification', color: 'from-teal-600 to-emerald-600', tag: 'REG/CLF' },
  { title: 'Random Forest', desc: 'Bagging, feature importance, OOB error, and K-fold cross-validation.', icon: '🌲', path: '/random-forest/regression', color: 'from-emerald-600 to-violet-600', tag: 'REG/CLF' },
]

export default function Home() {
  const scrollTo = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="animate-slide-up pb-20">
      
      {/* Hero Section */}
      <div className="mb-16 text-center pt-8">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-xs font-medium mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          Interactive DS Fundamentals
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight mb-4">
          <span className="bg-gradient-to-r from-violet-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Data Science & ML
          </span>
          <br />
          <span className="text-white/90">Education Platform</span>
        </h1>
        <p className="text-lg text-white/40 max-w-2xl mx-auto leading-relaxed mb-8">
          Master the foundation of Machine Learning. Explore theory, understand the pipeline, and experiment with real models.
        </p>

        {/* Anchor Links */}
        <div className="flex flex-wrap justify-center gap-3">
          <button onClick={() => scrollTo('ds-vs-da')} className="px-4 py-2 rounded-lg bg-white/[0.04] border border-white/[0.06] text-sm text-white/70 hover:bg-violet-600/20 hover:border-violet-500/30 transition-all font-medium flex items-center gap-2">
            <span>📊</span> DS vs Analytics
          </button>
          <button onClick={() => scrollTo('learning-types')} className="px-4 py-2 rounded-lg bg-white/[0.04] border border-white/[0.06] text-sm text-white/70 hover:bg-blue-600/20 hover:border-blue-500/30 transition-all font-medium flex items-center gap-2">
            <span>🧠</span> Learning Types
          </button>
          <button onClick={() => scrollTo('lifecycle')} className="px-4 py-2 rounded-lg bg-white/[0.04] border border-white/[0.06] text-sm text-white/70 hover:bg-emerald-600/20 hover:border-emerald-500/30 transition-all font-medium flex items-center gap-2">
            <span>⚙️</span> The Pipeline
          </button>
          <button onClick={() => scrollTo('labs')} className="px-4 py-2 rounded-lg bg-violet-600 text-sm text-white shadow-glow hover:bg-violet-500 transition-all font-medium flex items-center gap-2">
            <span>🕹️</span> Model Labs
          </button>
        </div>
      </div>

      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent mb-16" />

      {/* Section 1: DS vs DA */}
      <section id="ds-vs-da" className="mb-24 scroll-mt-24">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <span className="text-3xl">📊</span> Data Science vs. Data Analytics
        </h2>
        <p className="text-white/50 text-sm mb-8 leading-relaxed max-w-4xl">
          While often used interchangeably, these two fields answer entirely different questions about data. Analytics looks at the past, while Science predicts the future.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="glass-card p-8 border-t-2 border-t-cyan-500">
            <h3 className="text-xl font-semibold text-cyan-400 mb-4">Data Analytics</h3>
            <p className="text-white/60 text-sm mb-4">Focuses on processing and performing statistical analysis on existing datasets. Answers "What happened?" and "Why did it happen?".</p>
            <ul className="space-y-3 test-sm text-white/50">
              <li className="flex items-start gap-2"><span className="text-cyan-500">✓</span> <strong>Tools:</strong> SQL, Excel, Tableau, PowerBI</li>
              <li className="flex items-start gap-2"><span className="text-cyan-500">✓</span> <strong>Output:</strong> Dashboards, KPIs, retrospective reports</li>
              <li className="flex items-start gap-2"><span className="text-cyan-500">✓</span> <strong>Scope:</strong> Descriptive and Diagnostic</li>
            </ul>
          </div>
          <div className="glass-card p-8 border-t-2 border-t-violet-500">
            <h3 className="text-xl font-semibold text-violet-400 mb-4">Data Science</h3>
            <p className="text-white/60 text-sm mb-4">An interdisciplinary field utilizing ML and algorithms to extract knowledge. Answers "What will happen next?" and "What should we do?".</p>
            <ul className="space-y-3 test-sm text-white/50">
              <li className="flex items-start gap-2"><span className="text-violet-500">✓</span> <strong>Tools:</strong> Python, Scikit-Learn, TensorFlow, PyTorch</li>
              <li className="flex items-start gap-2"><span className="text-violet-500">✓</span> <strong>Output:</strong> Predictive models, automated recommendation engines</li>
              <li className="flex items-start gap-2"><span className="text-violet-500">✓</span> <strong>Scope:</strong> Predictive and Prescriptive</li>
            </ul>
          </div>
        </div>
      </section>

      {/* Section 2: Learning Types */}
      <section id="learning-types" className="mb-24 scroll-mt-24">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <span className="text-3xl">🧠</span> Core Machine Learning Types
        </h2>
        <p className="text-white/50 text-sm mb-8 leading-relaxed max-w-4xl">
          At the heart of Data Science is Machine Learning (ML). Almost all modern ML algorithms fall into two primary categories depending on the nature of the data they are fed.
        </p>
        
        <SupervisedDiagram />

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
          <div className="glass-card p-6">
            <h4 className="font-semibold text-white/90 mb-2">1. Supervised Learning</h4>
            <p className="text-sm text-white/50 mb-4">The algorithm is trained on a labeled dataset. It knows the "answer key" during training and attempts to learn the relationship.</p>
            <div className="bg-white/5 rounded-lg p-3 text-xs">
              <strong>Regression:</strong> Predicting a continuous number (e.g., house price).<br/>
              <strong>Classification:</strong> Predicting a specific category (e.g., Cat vs Dog).
            </div>
          </div>
          <div className="glass-card p-6">
            <h4 className="font-semibold text-white/90 mb-2">2. Unsupervised Learning</h4>
            <p className="text-sm text-white/50 mb-4">The algorithm is given data without explicit instructions or labels. It must discover the inherent structure automatically.</p>
            <div className="bg-white/5 rounded-lg p-3 text-xs">
              <strong>Clustering:</strong> Grouping similar data points together (e.g., customer segmentation).<br/>
              <strong>Dimensionality Reduction:</strong> Compressing features while retaining information.
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: The Pipeline */}
      <section id="lifecycle" className="mb-24 scroll-mt-24">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <span className="text-3xl">⚙️</span> The ML Lifecycle
        </h2>
        <p className="text-white/50 text-sm mb-8 leading-relaxed max-w-4xl">
          Building a model isn't just writing `model.fit()`. Real-world Data Science requires a rigorous, cyclical pipeline to ensure predictions are robust and unbiased.
        </p>
        
        <MLFlowchart />
      </section>

      <div className="w-full h-px bg-gradient-to-r from-transparent via-white/10 to-transparent mb-16" />

      {/* Section 4: Interactive Modules */}
      <section id="labs" className="scroll-mt-24">
        <div className="text-center mb-10">
          <h2 className="text-3xl font-bold text-white mb-4">Interactive Model Labs</h2>
          <p className="text-white/40 text-sm">Step outside the theory and get your hands dirty. Tweak parameters, visualize boundaries, and learn.</p>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
          {MODULES.map(m => (
            <NavLink key={m.path} to={m.path} className="group block">
              <div className="glass-card gradient-border p-6 h-full transition-all duration-300 group-hover:-translate-y-1 group-hover:shadow-glow">
                <div className={`w-10 h-10 rounded-xl bg-gradient-to-br ${m.color} flex items-center justify-center text-xl mb-4 shadow-glow`}>
                  {m.icon}
                </div>
                <div className="flex items-start justify-between mb-2">
                  <h2 className="text-base font-semibold text-white/90">{m.title}</h2>
                  <span className="badge badge-regression text-[9px] mt-0.5">{m.tag}</span>
                </div>
                <p className="text-sm text-white/40 leading-relaxed">{m.desc}</p>
                <div className="mt-4 flex items-center gap-1 text-xs text-violet-400 font-medium group-hover:translate-x-1 transition-transform">
                  Enter lab <span>→</span>
                </div>
              </div>
            </NavLink>
          ))}
        </div>
      </section>

    </div>
  )
}
