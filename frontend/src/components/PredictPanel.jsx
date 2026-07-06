export default function PredictPanel({
  teams, home, away, neutral,
  setHome, setAway, setNeutral,
  onPredict, onAnalysis, loading,
}) {
  return (
    <div style={{
      background: 'var(--surface)', border: '1px solid var(--border)',
      borderRadius: 14, padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1.2rem',
    }}>
      {/* Team selectors */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '0.75rem', alignItems: 'end' }}>
        <label style={{ display:'flex', flexDirection:'column', gap:6, fontSize:'0.85rem', color:'var(--muted)', fontWeight:600, letterSpacing:'0.05em', textTransform:'uppercase' }}>
          Home / Team 1
          <select value={home} onChange={e => setHome(e.target.value)} disabled={loading}>
            {teams.map(t => <option key={t}>{t}</option>)}
          </select>
        </label>

        <span style={{ fontSize:'1.4rem', paddingBottom:8, color:'var(--muted)' }}>vs</span>

        <label style={{ display:'flex', flexDirection:'column', gap:6, fontSize:'0.85rem', color:'var(--muted)', fontWeight:600, letterSpacing:'0.05em', textTransform:'uppercase' }}>
          Away / Team 2
          <select value={away} onChange={e => setAway(e.target.value)} disabled={loading}>
            {teams.map(t => <option key={t}>{t}</option>)}
          </select>
        </label>
      </div>

      {/* Neutral venue toggle */}
      <label style={{ display:'flex', alignItems:'center', gap:10, cursor:'pointer', userSelect:'none', color:'var(--muted)', fontSize:'0.9rem' }}>
        <input
          type="checkbox" checked={neutral} onChange={e => setNeutral(e.target.checked)}
          style={{ width:18, height:18, accentColor:'var(--green)', cursor:'pointer' }}
          disabled={loading}
        />
        Neutral venue (World Cup tournament setting)
      </label>

      {/* Action buttons */}
      <div style={{ display:'flex', gap:'0.75rem' }}>
        <button
          onClick={onPredict} disabled={loading}
          style={{ flex:1, padding:'0.75rem', fontSize:'1rem', background:'var(--green)', color:'#fff' }}
        >
          {loading ? '⏳ Predicting…' : '⚽ Predict Outcome'}
        </button>
        <button
          onClick={onAnalysis} disabled={loading}
          style={{ flex:1, padding:'0.75rem', fontSize:'1rem', background:'var(--gold)', color:'#000' }}
        >
          {loading ? '⏳ Analysing…' : '🔍 Get AI Preview'}
        </button>
      </div>
    </div>
  )
}
