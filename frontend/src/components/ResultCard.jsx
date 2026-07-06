function ProbBar({ label, value, color }) {
  const pct = (value * 100).toFixed(1)
  return (
    <div style={{ display:'flex', flexDirection:'column', gap:4 }}>
      <div style={{ display:'flex', justifyContent:'space-between', fontSize:'0.85rem' }}>
        <span style={{ color:'var(--muted)', fontWeight:600 }}>{label}</span>
        <span style={{ fontWeight:700, color }}>{pct}%</span>
      </div>
      <div style={{ height:12, borderRadius:99, background:'var(--border)', overflow:'hidden' }}>
        <div style={{
          height:'100%', width:`${pct}%`, background: color,
          borderRadius:99, transition:'width 0.6s ease',
        }} />
      </div>
    </div>
  )
}

export default function ResultCard({ result, preview, home, away }) {
  const p = result.probabilities
  const model = result.model || 'phase1'
  const badge = model === 'elo_baseline'
    ? { label:'Elo baseline', bg:'#1e3a5f', color:'#93c5fd' }
    : { label:'Poisson + XGBoost', bg:'#14532d', color:'#86efac' }

  return (
    <div style={{
      background:'var(--surface)', border:'1px solid var(--border)',
      borderRadius:14, padding:'1.5rem', marginTop:'1.5rem',
      display:'flex', flexDirection:'column', gap:'1.2rem',
    }}>
      {/* Match header */}
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <h2 style={{ fontSize:'1.15rem', fontWeight:700 }}>
          {home} <span style={{ color:'var(--muted)' }}>vs</span> {away}
        </h2>
        <span style={{
          fontSize:'0.72rem', fontWeight:700, letterSpacing:'0.05em', textTransform:'uppercase',
          padding:'3px 10px', borderRadius:99, background: badge.bg, color: badge.color,
        }}>
          {badge.label}
        </span>
      </div>

      {/* Probability bars */}
      <div style={{ display:'flex', flexDirection:'column', gap:'0.9rem' }}>
        <ProbBar label={`${home} Win`}  value={p.win}  color="var(--win)" />
        <ProbBar label="Draw"           value={p.draw} color="var(--draw)" />
        <ProbBar label={`${away} Win`}  value={p.loss} color="var(--loss)" />
      </div>

      {/* Most likely outcome highlight */}
      {(() => {
        const entries = [
          { label: `${home} Win`, v: p.win,  color:'var(--win)' },
          { label: 'Draw',        v: p.draw, color:'var(--draw)' },
          { label: `${away} Win`, v: p.loss, color:'var(--loss)' },
        ]
        const best = entries.reduce((a, b) => a.v > b.v ? a : b)
        return (
          <div style={{
            background:'var(--bg)', border:'1px solid var(--border)',
            borderRadius:10, padding:'0.75rem 1rem',
            display:'flex', alignItems:'center', gap:10, fontSize:'0.9rem',
          }}>
            <span style={{ color:'var(--muted)' }}>Most likely:</span>
            <strong style={{ color: best.color }}>{best.label}</strong>
            <span style={{ color:'var(--muted)', marginLeft:'auto' }}>
              {(best.v * 100).toFixed(1)}%
            </span>
          </div>
        )
      })()}

      {/* AI preview */}
      {preview && (
        <div style={{
          borderTop:'1px solid var(--border)', paddingTop:'1rem',
          fontSize:'0.92rem', lineHeight:1.7, color:'var(--muted)',
        }}>
          <div style={{ display:'flex', alignItems:'center', gap:6, marginBottom:'0.5rem', color:'var(--text)', fontWeight:600 }}>
            🤖 AI Match Preview
            {result.prompt_tokens && (
              <span style={{ fontSize:'0.7rem', color:'var(--muted)', marginLeft:'auto' }}>
                {result.prompt_tokens} tokens
              </span>
            )}
          </div>
          {preview.startsWith('[') ? (
            <em style={{ color:'var(--muted)' }}>{preview}</em>
          ) : (
            <p>{preview}</p>
          )}
        </div>
      )}
    </div>
  )
}
