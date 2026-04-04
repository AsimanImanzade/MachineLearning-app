interface ConfusionMatrixProps {
  matrix: number[][]
  labels?: string[]
}

export default function ConfusionMatrix({ matrix, labels }: ConfusionMatrixProps) {
  const n = matrix.length
  const defaultLabels = labels ?? Array.from({ length: n }, (_, i) => `Class ${i}`)
  const max = Math.max(...matrix.flat())

  return (
    <div className="glass-card p-5">
      <h3 className="section-title mb-4">Confusion Matrix</h3>
      <div className="overflow-x-auto">
        <table className="mx-auto border-collapse">
          <thead>
            <tr>
              <th className="text-xs text-white/30 p-2 text-right">Actual ↓ / Pred →</th>
              {defaultLabels.map(l => (
                <th key={l} className="text-xs text-white/50 p-2 font-medium text-center">{l}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                <td className="text-xs text-white/50 pr-3 text-right font-medium">{defaultLabels[ri]}</td>
                {row.map((val, ci) => {
                  const isDiag = ri === ci
                  const intensity = max > 0 ? val / max : 0
                  const bg = isDiag
                    ? `rgba(124,58,237,${0.2 + intensity * 0.6})`
                    : `rgba(239,68,68,${intensity * 0.5})`
                  return (
                    <td key={ci} className="p-1">
                      <div
                        className="w-16 h-16 flex items-center justify-center rounded-lg text-sm font-bold font-mono transition-all"
                        style={{ background: bg, color: intensity > 0.5 ? 'white' : 'rgba(255,255,255,0.7)' }}
                      >
                        {val}
                      </div>
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <div className="flex gap-4 justify-center mt-3 text-xs text-white/30">
          <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-violet-500/60" />Correct (diagonal)</span>
          <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-red-500/40" />Errors</span>
        </div>
      </div>
    </div>
  )
}
