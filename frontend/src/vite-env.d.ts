/// <reference types="vite/client" />

declare module 'react-katex' {
  import { ReactNode, FC } from 'react'
  interface FormulaProps {
    math: string
    block?: boolean
    errorColor?: string
    renderError?: (error: Error | string) => ReactNode
  }
  export const InlineMath: FC<FormulaProps>
  export const BlockMath: FC<FormulaProps>
}

