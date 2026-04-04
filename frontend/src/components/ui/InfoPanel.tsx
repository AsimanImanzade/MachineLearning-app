interface InfoPanelProps {
  type?: 'info' | 'tip' | 'warn'
  title?: string
  children: React.ReactNode
}

const styles = {
  info: { border: 'border-l-violet-500', bg: 'bg-violet-500/5', icon: 'ℹ️', label: 'text-violet-400' },
  tip:  { border: 'border-l-cyan-500',   bg: 'bg-cyan-500/5',   icon: '💡', label: 'text-cyan-400' },
  warn: { border: 'border-l-amber-500',  bg: 'bg-amber-500/5',  icon: '⚠️', label: 'text-amber-400' },
}

export default function InfoPanel({ type = 'info', title, children }: InfoPanelProps) {
  const s = styles[type]
  return (
    <div className={`info-box border-l-4 ${s.border} ${s.bg} rounded-xl p-4 my-4`}>
      {title && (
        <div className={`flex items-center gap-1.5 text-xs font-semibold mb-2 ${s.label}`}>
          <span>{s.icon}</span> {title}
        </div>
      )}
      <div className="text-sm text-white/60 leading-relaxed">{children}</div>
    </div>
  )
}
