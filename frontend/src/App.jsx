import { useState } from 'react'
import PredictPanel from './components/PredictPanel.jsx'
import ResultCard from './components/ResultCard.jsx'

const TEAMS = [
  'Algeria','Argentina','Australia','Belgium','Bosnia and Herzegovina',
  'Brazil','Cameroon','Canada','Chile','Colombia','Costa Rica','Croatia',
  'Denmark','Ecuador','Egypt','England','France','Germany','Ghana','Greece',
  'Honduras','Iceland','Iran','Italy','Ivory Coast','Japan','Mexico','Morocco',
  'Netherlands','Nigeria','Panama','Peru','Poland','Portugal','Qatar','Russia',
  'Saudi Arabia','Senegal','Serbia','South Korea','Spain','Sweden','Switzerland',
  'Tunisia','United States','Uruguay','Wales',
]

export default function App() {
  const [home, setHome] = useState('Argentina')
  const [away, setAway] = useState('Brazil')
  const [neutral, setNeutral] = useState(true)
  const [result, setResult] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  async function handlePredict() {
    if (home === away) { setError('Pick different teams'); return }
    setLoading(true); setError(null); setResult(null); setPreview(null)
    try {
      const r = await fetch(`/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`)
      if (!r.ok) throw new Error(`API error ${r.status}`)
      setResult(await r.json())
    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  async function handleAnalysis() {
    if (home === away) { setError('Pick different teams'); return }
    setLoading(true); setError(null); setResult(null); setPreview(null)
    try {
      const r = await fetch(`/match/analysis?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}&neutral=${neutral}`)
      if (!r.ok) throw new Error(`API error ${r.status}`)
      const data = await r.json()
      setResult(data)
      setPreview(data.preview || null)
    } catch (e) { setError(e.message) }
    setLoading(false)
  }

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: '2rem 1rem' }}>
      {/* Header */}
      <header style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <div style={{ fontSize: '3rem' }}>⚽</div>
        <h1 style={{ fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.5px' }}>
          GoalEdge
        </h1>
        <p style={{ color: 'var(--muted)', marginTop: 4 }}>
          World Cup match predictor · Poisson + XGBoost + Elo blend
        </p>
      </header>

      <PredictPanel
        teams={TEAMS}
        home={home} away={away} neutral={neutral}
        setHome={setHome} setAway={setAway} setNeutral={setNeutral}
        onPredict={handlePredict} onAnalysis={handleAnalysis}
        loading={loading}
      />

      {error && (
        <div style={{ background:'#450a0a', border:'1px solid #ef4444', borderRadius:10, padding:'1rem', marginTop:'1.5rem', color:'#fca5a5' }}>
          {error}
        </div>
      )}

      {result && (
        <ResultCard result={result} preview={preview} home={home} away={away} />
      )}

      <footer style={{ textAlign:'center', marginTop:'3rem', color:'var(--muted)', fontSize:'0.8rem' }}>
        GoalEdge v1 · Elo + Phase-1 model · Probabilities are model estimates only
      </footer>
    </div>
  )
}
