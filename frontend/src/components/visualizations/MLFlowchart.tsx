export default function MLFlowchart() {
  const steps = [
    { title: '1. Collection', desc: 'Gather raw data from databases, APIs, or scraping.', color: 'from-blue-600 to-cyan-600' },
    { title: '2. Preparation', desc: 'Clean, impute missing values, and handle outliers.', color: 'from-cyan-600 to-teal-600' },
    { title: '3. EDA & Features', desc: 'Analyze distributions and engineer new variables.', color: 'from-teal-600 to-emerald-600' },
    { title: '4. Modeling', desc: 'Select algorithms, train, and tune hyperparameters.', color: 'from-emerald-600 to-violet-600' },
    { title: '5. Evaluation', desc: 'Test against unseen data using CV and key metrics.', color: 'from-violet-600 to-purple-600' },
    { title: '6. Deployment', desc: 'Serve the model via APIs (like FastAPI!) for use.', color: 'from-purple-600 to-pink-600' },
  ]

  return (
    <div className="flex flex-col md:flex-row gap-2 relative mt-4">
      {/* Background connector line */}
      <div className="absolute top-1/2 left-4 right-4 h-0.5 bg-white/5 -translate-y-1/2 hidden md:block" />
      
      {steps.map((s, i) => (
        <div key={i} className="flex-1 relative group">
          <div className="glass-card p-4 transition-all duration-300 hover:-translate-y-2 hover:shadow-glow relative z-10 h-full">
            <div className={`w-3 h-3 rounded-full bg-gradient-to-r ${s.color} mb-3`} />
            <h5 className="text-sm font-bold text-white/90 mb-1">{s.title}</h5>
            <p className="text-[10px] text-white/40 leading-relaxed">{s.desc}</p>
          </div>
        </div>
      ))}
    </div>
  )
}
