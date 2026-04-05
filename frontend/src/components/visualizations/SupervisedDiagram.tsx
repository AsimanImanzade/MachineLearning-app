export default function SupervisedDiagram() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Supervised */}
      <div className="bg-black/20 border border-white/10 rounded-xl p-4 flex flex-col items-center">
        <h4 className="text-white/80 font-medium text-sm mb-3">Supervised Learning</h4>
        <svg viewBox="0 0 200 120" className="w-full h-32 opacity-80">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.4)" />
            </marker>
          </defs>
          <g>
            <rect x="10" y="20" width="50" height="80" rx="4" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.2)" />
            <text x="35" y="45" fill="rgba(255,255,255,0.6)" fontSize="10" textAnchor="middle">Features(X)</text>
            <circle cx="25" cy="65" r="3" fill="#8b5cf6" />
            <circle cx="45" cy="75" r="3" fill="#06b6d4" />
            <circle cx="30" cy="85" r="3" fill="#8b5cf6" />
            
            <path d="M 65 60 L 95 60" stroke="rgba(255,255,255,0.4)" strokeWidth="2" markerEnd="url(#arrow)" />
            
            <rect x="105" y="40" width="40" height="40" rx="4" fill="rgba(139,92,246,0.2)" stroke="#8b5cf6" />
            <text x="125" y="64" fill="#a78bfa" fontSize="12" textAnchor="middle" fontWeight="bold">Model</text>

            <path d="M 150 60 L 170 60" stroke="rgba(255,255,255,0.4)" strokeWidth="2" markerEnd="url(#arrow)" />

            <text x="185" y="64" fill="#10b981" fontSize="12" textAnchor="middle" fontWeight="bold">y</text>
            <text x="185" y="78" fill="rgba(255,255,255,0.4)" fontSize="8" textAnchor="middle">Labels</text>
          </g>
        </svg>
        <p className="text-xs text-white/40 text-center mt-2 max-w-[200px]">Model learns the mapping between input data and known labels.</p>
      </div>

      {/* Unsupervised */}
      <div className="bg-black/20 border border-white/10 rounded-xl p-4 flex flex-col items-center">
        <h4 className="text-white/80 font-medium text-sm mb-3">Unsupervised Learning</h4>
        <svg viewBox="0 0 200 120" className="w-full h-32 opacity-80">
          <g>
            <rect x="10" y="20" width="50" height="80" rx="4" fill="rgba(255,255,255,0.05)" stroke="rgba(255,255,255,0.2)" />
            <text x="35" y="45" fill="rgba(255,255,255,0.6)" fontSize="10" textAnchor="middle">Data (X)</text>
            <circle cx="25" cy="65" r="3" fill="#9ca3af" />
            <circle cx="45" cy="75" r="3" fill="#9ca3af" />
            <circle cx="30" cy="85" r="3" fill="#9ca3af" />
            <circle cx="20" cy="75" r="3" fill="#9ca3af" />
            
            <path d="M 65 60 L 95 60" stroke="rgba(255,255,255,0.4)" strokeWidth="2" markerEnd="url(#arrow)" />
            
            <rect x="105" y="40" width="40" height="40" rx="4" fill="rgba(6,182,212,0.2)" stroke="#06b6d4" />
            <text x="125" y="64" fill="#67e8f9" fontSize="12" textAnchor="middle" fontWeight="bold">Model</text>

            <path d="M 150 60 L 170 60" stroke="rgba(255,255,255,0.4)" strokeWidth="2" markerEnd="url(#arrow)" />

            <circle cx="185" cy="55" r="8" fill="rgba(139,92,246,0.3)" stroke="#8b5cf6" strokeDasharray="2" />
            <circle cx="185" cy="55" r="2" fill="#8b5cf6" />
            <circle cx="185" cy="75" r="8" fill="rgba(16,185,129,0.3)" stroke="#10b981" strokeDasharray="2" />
            <circle cx="185" cy="75" r="2" fill="#10b981" />
            
            <text x="185" y="95" fill="rgba(255,255,255,0.4)" fontSize="8" textAnchor="middle">Clusters</text>
          </g>
        </svg>
        <p className="text-xs text-white/40 text-center mt-2 max-w-[200px]">Model finds hidden structures or patterns in unlabeled data.</p>
      </div>
    </div>
  )
}
