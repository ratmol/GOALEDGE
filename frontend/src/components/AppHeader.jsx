export default function AppHeader() {
  return (
    <header className="app-header">
      <div className="app-header__glow" aria-hidden="true" />
      <div style={{ position: 'relative', zIndex: 1 }}>
        <span className="app-header__label">GoalEdge · The Gaffer's Match Lab</span>
        <h1 className="app-header__headline">
          I don't pick winners.<br />
          I price the whole match.
        </h1>
        <p className="app-header__tagline">
          Goals, corners, cards, territory. Tell me the two sides
          and I'll show you how the 90 minutes breathe.
        </p>
        <p className="app-header__intro">
          Pick your teams below — get calibrated match probabilities,
          value odds, and The Gaffer's read on how it plays out.
        </p>
      </div>
    </header>
  )
}
