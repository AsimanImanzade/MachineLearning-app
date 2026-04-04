import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, ReferenceLine,
} from 'recharts'

interface OverfitPoint { degree?: number; depth?: number; k?: number; train_mse?: number; test_mse?: number; train_score?: number; test_score?: number }

interface OverfitCurveProps {
  data: OverfitPoint[]
  xKey: 'degree' | 'depth' | 'k'
  xLabel?: string
  isScore?: boolean  // true = higher is better (accuracy/R²), false = lower is better (MSE)
  currentX?: number
}

export default function OverfitCurve({ data, xKey, xLabel, isScore = false, currentX }: OverfitCurveProps) {
  const trainKey = isScore ? 'train_score' : 'train_mse'
  const testKey = isScore ? 'test_score' : 'test_mse'
  const trainLabel = isScore ? 'Train Score' : 'Train MSE'
  const testLabel = isScore ? 'Val Score' : 'Test MSE'

  const chartData = data.map(d => ({
    x: d[xKey],
    [trainLabel]: +(d[trainKey as keyof OverfitPoint] as number)?.toFixed(4),
    [testLabel]: +(d[testKey as keyof OverfitPoint] as number)?.toFixed(4),
  }))

  return (
    <div className="glass-card p-5">
      <h3 className="section-title mb-2">Bias–Variance Tradeoff</h3>
      <p className="text-xs text-white/40 mb-4">
        {isScore
          ? 'Gap between train and val score → overfitting. Both low → underfitting.'
          : 'Train MSE drops while Test MSE rises → overfitting. Both high → underfitting.'}
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart data={chartData} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="x" label={{ value: xLabel ?? xKey, position: 'insideBottom', offset: -5, fill: 'rgba(255,255,255,0.3)', fontSize: 11 }} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
          <YAxis tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} tickFormatter={v => v.toFixed(2)} />
          <Tooltip
            contentStyle={{ background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: '10px' }}
            itemStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '11px' }} />
          {currentX !== undefined && (
            <ReferenceLine x={currentX} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'Current', fill: '#f59e0b', fontSize: 10 }} />
          )}
          <Line type="monotone" dataKey={trainLabel} stroke="#7c3aed" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey={testLabel} stroke="#06b6d4" strokeWidth={2} dot={false} strokeDasharray="5 2" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
