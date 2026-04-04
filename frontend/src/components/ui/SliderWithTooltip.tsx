import { useState } from 'react'

interface SliderWithTooltipProps {
  label: string
  min: number
  max: number
  step: number
  value: number
  onChange: (v: number) => void
  tooltip: string
  formatValue?: (v: number) => string
  color?: string
}

export default function SliderWithTooltip({
  label, min, max, step, value, onChange, tooltip, formatValue, color = '#7c3aed',
}: SliderWithTooltipProps) {
  const [showTip, setShowTip] = useState(false)
  const pct = ((value - min) / (max - min)) * 100

  return (
    <div className="mb-5">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white/70">{label}</span>
          <button
            onMouseEnter={() => setShowTip(true)}
            onMouseLeave={() => setShowTip(false)}
            className="w-4 h-4 rounded-full border border-white/20 text-white/40 text-[10px] flex items-center justify-center hover:border-violet-400 hover:text-violet-400 transition-colors"
          >
            ?
          </button>
        </div>
        <span className="text-sm font-mono font-semibold text-violet-300">
          {formatValue ? formatValue(value) : value}
        </span>
      </div>

      {showTip && (
        <div className="glass-card p-3 mb-2 text-xs text-white/60 leading-relaxed border-l-2 border-violet-500/50 animate-fade-in">
          {tooltip}
        </div>
      )}

      <div className="relative h-5 flex items-center">
        {/* Track */}
        <div className="absolute w-full h-1.5 rounded-full bg-white/10" />
        {/* Fill */}
        <div
          className="absolute h-1.5 rounded-full transition-all"
          style={{ width: `${pct}%`, background: `linear-gradient(to right, ${color}, #2563eb)` }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={e => onChange(Number(e.target.value))}
          className="absolute w-full opacity-0 h-5 cursor-pointer z-10"
        />
        {/* Thumb */}
        <div
          className="absolute w-4 h-4 rounded-full bg-white border-2 shadow-glow pointer-events-none transition-all"
          style={{ left: `calc(${pct}% - 8px)`, borderColor: color, boxShadow: `0 0 12px ${color}80` }}
        />
      </div>
    </div>
  )
}
