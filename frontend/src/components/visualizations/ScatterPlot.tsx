import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Line, LineChart, ReferenceArea,
} from 'recharts'

interface ScatterPlotProps {
  trainX: number[]
  trainY: number[]
  testX: number[]
  testY: number[]
  trainPred?: number[]
  testPred?: number[]
  xLabel?: string
  yLabel?: string
}

export default function ScatterPlot({ trainX, trainY, testX, testY, trainPred, testPred, xLabel = 'X', yLabel = 'Y' }: ScatterPlotProps) {
  const trainData = trainX.map((x, i) => ({ x, y: trainY[i] }))
  const testData = testX.map((x, i) => ({ x, y: testY[i] }))

  // regression line through predictions sorted by x
  const regLine = trainPred
    ? trainX
        .map((x, i) => ({ x, y: trainPred[i] }))
        .sort((a, b) => a.x - b.x)
    : []

  return (
    <div className="glass-card p-4">
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
          <XAxis dataKey="x" name={xLabel} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} label={{ value: xLabel, position: 'insideBottom', offset: -10, fill: 'rgba(255,255,255,0.3)', fontSize: 11 }} />
          <YAxis dataKey="y" name={yLabel} tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 11 }} label={{ value: yLabel, angle: -90, position: 'insideLeft', fill: 'rgba(255,255,255,0.3)', fontSize: 11 }} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3', stroke: 'rgba(255,255,255,0.1)' }}
            contentStyle={{ background: 'rgba(13,13,26,0.95)', border: '1px solid rgba(124,58,237,0.3)', borderRadius: '10px' }}
            itemStyle={{ color: 'rgba(255,255,255,0.8)' }}
          />
          <Scatter name="Train" data={trainData} fill="#7c3aed" opacity={0.7} r={3} />
          <Scatter name="Test" data={testData} fill="#06b6d4" opacity={0.8} r={4} />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex gap-4 justify-center mt-2">
        <span className="text-xs text-white/40 flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-violet-500" /> Train</span>
        <span className="text-xs text-white/40 flex items-center gap-1"><span className="inline-block w-2 h-2 rounded-full bg-cyan-500" /> Test</span>
      </div>
    </div>
  )
}
