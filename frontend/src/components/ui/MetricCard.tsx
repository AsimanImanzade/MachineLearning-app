interface MetricCardProps {
  label: string
  value: number | string
  unit?: string
  color?: 'purple' | 'blue' | 'cyan' | 'green' | 'pink'
  description?: string
}

const gradients: Record<string, string> = {
  purple: 'from-violet-400 to-purple-400',
  blue: 'from-blue-400 to-cyan-400',
  cyan: 'from-cyan-400 to-teal-400',
  green: 'from-emerald-400 to-green-400',
  pink: 'from-pink-400 to-rose-400',
}

export default function MetricCard({ label, value, unit, color = 'purple', description }: MetricCardProps) {
  const grad = gradients[color]
  const display = typeof value === 'number' ? value.toFixed(4) : value

  return (
    <div className="metric-card gradient-border animate-slide-up">
      <div className="metric-label">{label}</div>
      <div className={`text-2xl font-bold font-mono bg-gradient-to-r ${grad} bg-clip-text text-transparent`}>
        {display}{unit && <span className="text-sm ml-1 opacity-60">{unit}</span>}
      </div>
      {description && (
        <div className="text-[11px] text-white/30 mt-1 leading-relaxed">{description}</div>
      )}
    </div>
  )
}
