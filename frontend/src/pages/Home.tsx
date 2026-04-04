import { NavLink } from 'react-router-dom'

const MODULES = [
  { title: 'Linear Regression', desc: 'OLS, Lasso, Ridge, ElasticNet — from the normal equation to regularization paths.', icon: '📈', path: '/linear/standard', color: 'from-violet-600 to-blue-600', tag: 'REG' },
  { title: 'Logistic Regression', desc: 'Sigmoid, log-loss, decision boundaries — probabilistic binary classification.', icon: '🔀', path: '/logistic', color: 'from-blue-600 to-cyan-600', tag: 'CLF' },
  { title: 'K-Nearest Neighbors', desc: 'Distance metrics, the k-overfitting curve, and Voronoi decision boundaries.', icon: '🎯', path: '/knn/classification', color: 'from-cyan-600 to-teal-600', tag: 'REG/CLF' },
  { title: 'Decision Trees', desc: 'Gini impurity, information gain, and why max_depth is your best friend.', icon: '🌳', path: '/decision-tree/classification', color: 'from-teal-600 to-emerald-600', tag: 'REG/CLF' },
  { title: 'Random Forest', desc: 'Bagging, feature importance, OOB error, and K-fold cross-validation.', icon: '🌲', path: '/random-forest/regression', color: 'from-emerald-600 to-violet-600', tag: 'REG/CLF' },
]

export default function Home() {
  return (
    <div className="animate-slide-up">
      {/* Hero */}
      <div className="mb-14 text-center">
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-violet-500/30 bg-violet-500/10 text-violet-300 text-xs font-medium mb-6">
          <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />
          FastAPI · React · Scikit-learn · KaTeX
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight mb-4">
          <span className="bg-gradient-to-r from-violet-400 via-blue-400 to-cyan-400 bg-clip-text text-transparent">
            Machine Learning
          </span>
          <br />
          <span className="text-white/90">Education Platform</span>
        </h1>
        <p className="text-lg text-white/40 max-w-2xl mx-auto leading-relaxed">
          Master ML algorithms through interactive experiments, real-time visualizations, 
          and rigorous mathematical explanations — not just black-box sliders.
        </p>
      </div>

      {/* Module cards */}
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
                Explore module <span>→</span>
              </div>
            </div>
          </NavLink>
        ))}
      </div>

      {/* Feature grid */}
      <div className="mt-12 grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { icon: '🧮', title: 'LaTeX Formulas', desc: 'Every metric and algorithm rendered with KaTeX' },
          { icon: '🎮', title: 'Overfit Sandbox', desc: 'Break the model intentionally and see why' },
          { icon: '🔬', title: 'Feature Engineering', desc: 'Drop columns, watch importance shift live' },
          { icon: '📊', title: 'K-Fold CV', desc: 'Visualize variance across folds' },
        ].map(f => (
          <div key={f.title} className="glass-card p-4 text-center">
            <div className="text-2xl mb-2">{f.icon}</div>
            <div className="text-xs font-semibold text-white/70 mb-1">{f.title}</div>
            <div className="text-[11px] text-white/30 leading-relaxed">{f.desc}</div>
          </div>
        ))}
      </div>
    </div>
  )
}
