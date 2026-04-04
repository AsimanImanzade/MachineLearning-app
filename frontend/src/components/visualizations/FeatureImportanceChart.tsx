import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts'

interface FeatureItem {
  feature: string
  importance: number
}

interface FeatureImportanceChartProps {
  data: FeatureItem[]
  activeFeatures?: Set<string>
}

const COLORS = ['#7c3aed','#6d28d9','#5b21b6','#4c1d95','#2563eb','#1d4ed8','#1e40af','#1e3a8a']

export default function FeatureImportanceChart({ data, activeFeatures }: FeatureImportanceChartProps) {
  const sorted = [...data].sort((a, b) => b.importance - a.importance)

  return (
    <div className="glass-card p-5">
      <h3 className="section-title mb-4">Feature Importances</h3>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={sorted} layout="vertical" margin={{ top: 0, right: 30, left: 80, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" horizontal={false} />
          <XAxis type="number" domain={[0, 1]} tickFormatter={v => (v * 100).toFixed(0) + '%'} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} />
          <YAxis dataKey="feature" type="category" tick={{ fill: 'rgba(255,255,255,0.6)', fontSize: 11 }} />
          <Tooltip
            formatter={(v: number) => [(v * 100).toFixed(2) + '%', 'Importance']}
            contentStyle={{ background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: '10px' }}
            itemStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
            {sorted.map((entry, i) => (
              <Cell
                key={entry.feature}
                fill={COLORS[i % COLORS.length]}
                opacity={!activeFeatures || activeFeatures.has(entry.feature) ? 1 : 0.2}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}
