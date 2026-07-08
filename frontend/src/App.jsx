import { useState } from 'react'
import AppHeader from './components/AppHeader.jsx'
import EmptyState from './components/EmptyState.jsx'
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
    <>
      <AppHeader />
      <main className="app-shell">
        <PredictPanel
          teams={TEAMS}
          home={home} away={away} neutral={neutral}
          setHome={setHome} setAway={setAway} setNeutral={setNeutral}
          onPredict={handlePredict} onAnalysis={handleAnalysis}
          loading={loading}
        />

        {error && (
          <div className="error-box" role="alert">{error}</div>
        )}

        {!result && !error && !loading && <EmptyState />}

        <div aria-live="polite" aria-atomic="false">
        {loading && !result && (
          <div className="result-card" aria-busy="true" aria-label="Loading analysis">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 }}>
              <div className="skeleton" style={{ height: 18, width: '50%', borderRadius: 4 }} />
              <div className="skeleton" style={{ height: 18, width: 110, borderRadius: 99 }} />
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div className="skeleton" style={{ height: 8, borderRadius: 99 }} />
              <div className="skeleton" style={{ height: 8, width: '80%', borderRadius: 99 }} />
              <div className="skeleton" style={{ height: 8, width: '65%', borderRadius: 99 }} />
            </div>
          </div>
        )}

        {result && (
          <ResultCard result={result} preview={preview} home={home} away={away} />
        )}
        </div>

        <footer className="app-footer">
          GoalEdge · Elo + Poisson + XGBoost blend · Probabilities are model estimates only
        </footer>
      </main>
    </>
  )
}
