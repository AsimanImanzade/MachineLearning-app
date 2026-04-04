import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine, Legend,
} from 'recharts'

interface Fold {
  fold: number
  train_score: number
  val_score: number
}

interface CVFoldChartProps {
  folds: Fold[]
  metric?: string
}

export default function CVFoldChart({ folds, metric = 'R² Score' }: CVFoldChartProps) {
  const mean = folds.reduce((s, f) => s + f.val_score, 0) / folds.length
  const variance = folds.reduce((s, f) => s + Math.pow(f.val_score - mean, 2), 0) / folds.length
  const std = Math.sqrt(variance)

  const data = folds.map(f => ({
    name: `Fold ${f.fold}`,
    'Train Score': +f.train_score.toFixed(4),
    'Val Score': +f.val_score.toFixed(4),
  }))

  return (
    <div className="glass-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="section-title">Cross-Validation Results</h3>
        <div className="flex gap-4">
          <div className="text-center">
            <div className="text-xs text-white/30 uppercase tracking-wider">Mean Val {metric}</div>
            <div className="text-lg font-bold font-mono text-violet-300">{mean.toFixed(4)}</div>
          </div>
          <div className="text-center">
            <div className="text-xs text-white/30 uppercase tracking-wider">Std Dev</div>
            <div className="text-lg font-bold font-mono text-amber-300">{std.toFixed(4)}</div>
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} barGap={4} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
          <YAxis domain={[0, 1]} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: '10px' }}
            itemStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Legend wrapperStyle={{ paddingTop: '12px', fontSize: '11px' }} />
          <ReferenceLine y={mean} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: `Mean=${mean.toFixed(3)}`, fill: '#f59e0b', fontSize: 10 }} />
          <Bar dataKey="Train Score" fill="#7c3aed" opacity={0.6} radius={[4,4,0,0]} />
          <Bar dataKey="Val Score" radius={[4,4,0,0]}>
            {data.map((_, i) => (
              <Cell key={i} fill={`hsl(${210 + i * 20}, 80%, 60%)`} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 p-3 bg-amber-500/5 border border-amber-500/20 rounded-xl text-xs text-amber-300/80">
        <strong>Why does variance matter?</strong> A high standard deviation ({std.toFixed(4)}) across folds means your model's performance depends heavily on which data it sees — a sign of instability. Single train/test splits can mask this completely.
      </div>
    </div>
  )
}
