export default function EmptyState() {
  return (
    <div className="empty-state" role="region" aria-label="No prediction yet">
      <blockquote className="empty-state__quote">
        "A prediction without a 'why' is a coin flip with a logo."
      </blockquote>
      <div>
        <p className="empty-state__label">What you'll see</p>
        <div className="empty-state__modes" style={{ marginTop: '0.5rem' }}>
          <span>Match probabilities</span>
          <span className="empty-state__dot" aria-hidden="true">·</span>
          <span>Calibrated odds</span>
          <span className="empty-state__dot" aria-hidden="true">·</span>
          <span>The Gaffer's read</span>
        </div>
      </div>
    </div>
  )
}
