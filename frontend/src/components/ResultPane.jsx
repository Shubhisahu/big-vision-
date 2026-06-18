import './ResultPane.css'

function ConfBar({ value, colour }) {
  const pct = Math.round(value * 100)
  return (
    <div className="conf-bar-wrap">
      <div className="conf-bar-track">
        <div
          className="conf-bar-fill"
          style={{ width: `${pct}%`, background: colour }}
        />
      </div>
      <span className="conf-value" style={{ color: colour }}>{pct}%</span>
    </div>
  )
}

function DetectionCard({ det, index, colour }) {
  const { class_name, confidence, bbox } = det
  const hex = colour || '#6366f1'

  return (
    <div className="det-card glass-card" style={{ '--cls-colour': hex }}>
      <div className="det-header">
        <div className="det-index">#{index + 1}</div>
        <div className="det-class-badge" style={{ background: `${hex}22`, borderColor: `${hex}55`, color: hex }}>
          {class_name}
        </div>
        {det.track_id != null && (
          <div className="det-track-badge">ID {det.track_id}</div>
        )}
      </div>

      <div className="det-conf-row">
        <span className="det-label">Confidence</span>
        <ConfBar value={confidence} colour={hex} />
      </div>

      {bbox && (
        <div className="det-bbox">
          <span className="det-label">Bounding Box</span>
          <div className="bbox-grid">
            <div className="bbox-coord"><span>x1</span><strong>{Math.round(bbox.x1)}</strong></div>
            <div className="bbox-coord"><span>y1</span><strong>{Math.round(bbox.y1)}</strong></div>
            <div className="bbox-coord"><span>x2</span><strong>{Math.round(bbox.x2)}</strong></div>
            <div className="bbox-coord"><span>y2</span><strong>{Math.round(bbox.y2)}</strong></div>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ResultPane({ result, classColours }) {
  if (!result) return null

  const { count, by_class, inference_ms, detections = [] } = result

  return (
    <div className="result-root">
      {/* Summary stats */}
      <div className="result-summary glass-card">
        <div className="stat-block">
          <span className="stat-value">{count}</span>
          <span className="stat-label">Total Objects</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-block">
          <span className="stat-value">{inference_ms != null ? `${inference_ms.toFixed(1)}ms` : '—'}</span>
          <span className="stat-label">Inference Time</span>
        </div>
        <div className="stat-divider" />
        <div className="stat-block">
          <span className="stat-value">{Object.keys(by_class || {}).length}</span>
          <span className="stat-label">Classes Found</span>
        </div>
      </div>

      {/* By-class badges */}
      {by_class && Object.keys(by_class).length > 0 && (
        <div className="by-class-row">
          {Object.entries(by_class).map(([cls, cnt]) => {
            const colour = classColours?.[cls] || '#6366f1'
            return (
              <div
                key={cls}
                className="class-badge"
                style={{ background: `${colour}18`, borderColor: `${colour}50`, color: colour }}
              >
                <span
                  className="class-badge-dot"
                  style={{ background: colour, boxShadow: `0 0 6px ${colour}` }}
                />
                <span className="class-badge-name">{cls}</span>
                <span className="class-badge-count">{cnt}</span>
              </div>
            )
          })}
        </div>
      )}

      {/* Detection cards */}
      <div className="det-list">
        {detections.length === 0 ? (
          <div className="no-detections">
            <span>🔍</span>
            <p>No detections above threshold</p>
          </div>
        ) : (
          detections.map((det, i) => (
            <DetectionCard
              key={i}
              det={det}
              index={i}
              colour={det.colour_hex || classColours?.[det.class_name]}
            />
          ))
        )}
      </div>
    </div>
  )
}
