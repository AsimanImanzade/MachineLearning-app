import { useMemo } from 'react'

interface BoundaryData {
  xx: number[][]
  yy: number[][]
  zz: number[][]
  x_train: number[]
  y_train: number[]
  labels_train: number[]
  x_test: number[]
  y_test: number[]
  labels_test: number[]
}

interface DecisionBoundaryProps {
  data: BoundaryData
  width?: number
  height?: number
}

const CLASS_COLORS = ['#7c3aed', '#06b6d4', '#f59e0b', '#10b981']
const CLASS_BG = ['rgba(124,58,237,0.25)', 'rgba(6,182,212,0.25)', 'rgba(245,158,11,0.25)', 'rgba(16,185,129,0.25)']

export default function DecisionBoundary({ data, width = 480, height = 380 }: DecisionBoundaryProps) {
  const { xx, yy, zz, x_train, y_train, labels_train, x_test, y_test, labels_test } = data

  const rows = zz.length
  const cols = zz[0]?.length ?? 0

  const xMin = xx[0][0], xMax = xx[0][cols - 1]
  const yMin = yy[0][0], yMax = yy[rows - 1][0]

  const toSvg = (px: number, py: number) => ({
    cx: ((px - xMin) / (xMax - xMin)) * width,
    cy: height - ((py - yMin) / (yMax - yMin)) * height,
  })

  // Build colored pixel grid as canvas-like approach using SVG rects
  const cellW = width / cols
  const cellH = height / rows

  const cells = useMemo(() => {
    const arr: { x: number; y: number; cls: number }[] = []
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        arr.push({ x: c * cellW, y: r * cellH, cls: zz[rows - 1 - r][c] })
      }
    }
    return arr
  }, [zz, rows, cols, cellW, cellH])

  return (
    <div className="glass-card p-4 overflow-hidden">
      <h3 className="section-title mb-3">Decision Boundary</h3>
      <svg width={width} height={height} className="rounded-xl" style={{ maxWidth: '100%' }}>
        {/* Background boundary cells */}
        {cells.map((cell, i) => (
          <rect
            key={i}
            x={cell.x}
            y={cell.y}
            width={cellW + 1}
            height={cellH + 1}
            fill={CLASS_BG[cell.cls % CLASS_BG.length]}
          />
        ))}

        {/* Train points */}
        {x_train.map((px, i) => {
          const { cx, cy } = toSvg(px, y_train[i])
          return (
            <circle key={`tr${i}`} cx={cx} cy={cy} r={4}
              fill={CLASS_COLORS[labels_train[i] % CLASS_COLORS.length]}
              opacity={0.8} stroke="rgba(0,0,0,0.3)" strokeWidth={0.5} />
          )
        })}

        {/* Test points – larger, outlined */}
        {x_test.map((px, i) => {
          const { cx, cy } = toSvg(px, y_test[i])
          return (
            <circle key={`te${i}`} cx={cx} cy={cy} r={6}
              fill={CLASS_COLORS[labels_test[i] % CLASS_COLORS.length]}
              opacity={0.95} stroke="white" strokeWidth={1.5} />
          )
        })}
      </svg>
      <div className="flex gap-4 mt-2 text-xs text-white/40">
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-white/50 inline-block" /> Test (outlined)</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-white/30 inline-block" /> Train (filled)</span>
      </div>
    </div>
  )
}
