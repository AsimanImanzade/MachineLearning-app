import { NavLink, useLocation } from 'react-router-dom'
import { useState } from 'react'

const NAV = [
  {
    group: 'Linear Regression',
    icon: '📈',
    items: [
      { label: 'Standard (OLS)', path: '/linear/standard', tag: 'reg' },
      { label: 'Lasso (L1)', path: '/linear/lasso', tag: 'reg' },
      { label: 'Ridge (L2)', path: '/linear/ridge', tag: 'reg' },
      { label: 'ElasticNet', path: '/linear/elasticnet', tag: 'reg' },
    ],
  },
  {
    group: 'Logistic Regression',
    icon: '🔀',
    items: [
      { label: 'Binary Classification', path: '/logistic', tag: 'clf' },
    ],
  },
  {
    group: 'K-Nearest Neighbors',
    icon: '🎯',
    items: [
      { label: 'Classification', path: '/knn/classification', tag: 'clf' },
      { label: 'Regression', path: '/knn/regression', tag: 'reg' },
    ],
  },
  {
    group: 'Decision Trees',
    icon: '🌳',
    items: [
      { label: 'Classification', path: '/decision-tree/classification', tag: 'clf' },
      { label: 'Regression', path: '/decision-tree/regression', tag: 'reg' },
    ],
  },
  {
    group: 'Random Forest',
    icon: '🌲',
    items: [
      { label: 'Regression', path: '/random-forest/regression', tag: 'reg' },
      { label: 'Classification', path: '/random-forest/classification', tag: 'clf' },
    ],
  },
]

export default function Sidebar() {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  const toggle = (group: string) =>
    setCollapsed(prev => ({ ...prev, [group]: !prev[group] }))

  return (
    <aside className="fixed left-0 top-0 h-screen w-64 flex flex-col z-40 border-r border-white/[0.06] bg-base-800/80 backdrop-blur-xl">
      {/* Logo */}
      <div className="flex items-center gap-3 px-5 py-5 border-b border-white/[0.06]">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-500 to-blue-500 flex items-center justify-center text-white font-bold text-sm shadow-glow">
          DS
        </div>
        <div>
          <div className="text-sm font-semibold text-white">DataScienceApp</div>
          <div className="text-[10px] text-white/30 font-mono">ML Education Platform</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 overflow-y-auto px-3 py-4 space-y-1">
        <NavLink
          to="/"
          className={({ isActive }) =>
            `nav-item ${isActive ? 'active' : ''}`
          }
        >
          <span>🏠</span> Home
        </NavLink>

        <div className="my-3 border-t border-white/[0.06]" />

        {NAV.map(({ group, icon, items }) => (
          <div key={group} className="mb-1">
            <button
              onClick={() => toggle(group)}
              className="w-full flex items-center justify-between px-3 py-2 rounded-lg text-xs font-semibold text-white/30 uppercase tracking-widest hover:text-white/60 transition-colors"
            >
              <span className="flex items-center gap-2">
                <span>{icon}</span> {group}
              </span>
              <span className="text-white/20">{collapsed[group] ? '▶' : '▼'}</span>
            </button>

            {!collapsed[group] && (
              <div className="ml-2 space-y-0.5">
                {items.map(({ label, path, tag }) => (
                  <NavLink
                    key={path}
                    to={path}
                    className={({ isActive }) =>
                      `nav-item ${isActive ? 'active' : ''}`
                    }
                  >
                    <span className="flex-1 text-xs">{label}</span>
                    <span className={`badge ${tag === 'reg' ? 'badge-regression' : 'badge-classification'}`}>
                      {tag === 'reg' ? 'REG' : 'CLF'}
                    </span>
                  </NavLink>
                ))}
              </div>
            )}
          </div>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-white/[0.06]">
        <div className="text-[10px] text-white/20 font-mono">FastAPI + React + Scikit-learn</div>
      </div>
    </aside>
  )
}
