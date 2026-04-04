import 'katex/dist/katex.min.css'
import { InlineMath, BlockMath } from 'react-katex'

interface FormulaBlockProps {
  formula: string
  label?: string
  inline?: boolean
}

export default function FormulaBlock({ formula, label, inline = false }: FormulaBlockProps) {
  if (inline) return <InlineMath math={formula} />

  return (
    <div className="formula-block">
      {label && (
        <div className="text-xs font-mono text-violet-400 mb-3 uppercase tracking-widest">{label}</div>
      )}
      <BlockMath math={formula} />
    </div>
  )
}
