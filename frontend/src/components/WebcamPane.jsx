import { useState, useRef, useEffect, useCallback } from 'react'
import ResultPane from './ResultPane.jsx'
import './WebcamPane.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const FRAME_INTERVAL = 200 // ms

export default function WebcamPane({ classColours }) {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const intervalRef = useRef(null)
  const lastResultRef = useRef(null)

  const [isRunning, setIsRunning] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [fps, setFps] = useState(0)
  const [frameCount, setFrameCount] = useState(0)
  const fpsFrameRef = useRef(0)
  const fpsTimeRef = useRef(performance.now())
  const sessionId = useRef(`session_${Date.now()}`)

  // Draw detections on canvas overlay
  const drawDetections = useCallback((detections, videoW, videoH, canvasW, canvasH) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, canvasW, canvasH)

    const scaleX = canvasW / videoW
    const scaleY = canvasH / videoH

    detections.forEach((det) => {
      const { bbox, class_name, confidence, colour_hex, track_id } = det
      if (!bbox) return
      const colour = colour_hex || classColours?.[class_name] || '#6366f1'

      const x = bbox.x1 * scaleX
      const y = bbox.y1 * scaleY
      const w = (bbox.x2 - bbox.x1) * scaleX
      const h = (bbox.y2 - bbox.y1) * scaleY

      // Box glow effect
      ctx.shadowColor = colour
      ctx.shadowBlur = 12

      // Border
      ctx.strokeStyle = colour
      ctx.lineWidth = 2
      ctx.strokeRect(x, y, w, h)

      // Corner marks
      const cl = 14
      ctx.lineWidth = 3
      ctx.beginPath()
      ;[
        [x, y + cl, x, y, x + cl, y],
        [x + w - cl, y, x + w, y, x + w, y + cl],
        [x + w, y + h - cl, x + w, y + h, x + w - cl, y + h],
        [x + cl, y + h, x, y + h, x, y + h - cl],
      ].forEach(([x1, y1, x2, y2, x3, y3]) => {
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.lineTo(x3, y3)
      })
      ctx.stroke()

      ctx.shadowBlur = 0

      // Label background
      const label = `${class_name}${track_id != null ? ` #${track_id}` : ''} ${Math.round(confidence * 100)}%`
      ctx.font = '600 11px Inter, sans-serif'
      const textWidth = ctx.measureText(label).width
      const labelH = 20
      const labelY = y > labelH + 4 ? y - labelH - 4 : y + 2
      ctx.fillStyle = colour + 'dd'
      ctx.beginPath()
      ctx.roundRect(x - 1, labelY, textWidth + 14, labelH, 4)
      ctx.fill()
      ctx.fillStyle = '#000'
      ctx.fillText(label, x + 6, labelY + 13)
    })
  }, [classColours])

  // Capture and send frame
  const captureAndSend = useCallback(async () => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    const vw = video.videoWidth
    const vh = video.videoHeight
    if (!vw || !vh) return

    // Sync canvas size
    if (canvas.width !== vw || canvas.height !== vh) {
      canvas.width = vw
      canvas.height = vh
    }

    // Encode frame as JPEG blob
    const offscreen = document.createElement('canvas')
    offscreen.width = vw
    offscreen.height = vh
    const oc = offscreen.getContext('2d')
    oc.drawImage(video, 0, 0, vw, vh)

    offscreen.toBlob(async (blob) => {
      if (!blob) return
      try {
        const fd = new FormData()
        fd.append('file', blob, 'frame.jpg')

        const res = await fetch(`${API_BASE}/infer-frame?session_id=${sessionId.current}`, { method: 'POST', body: fd })
        if (!res.ok) {
          console.error(`HTTP request to /infer-frame failed with status ${res.status}`)
          return
        }

        const data = await res.json()
        lastResultRef.current = data
        setResult(data)
        setFrameCount((c) => c + 1)

        // Draw overlays
        drawDetections(data.detections || [], vw, vh, canvas.width, canvas.height)

        // FPS update
        fpsFrameRef.current++
        const now = performance.now()
        const elapsed = now - fpsTimeRef.current
        if (elapsed >= 1000) {
          setFps(Math.round((fpsFrameRef.current * 1000) / elapsed))
          fpsFrameRef.current = 0
          fpsTimeRef.current = now
        }
      } catch (e) {
        console.error(`HTTP request to /infer-frame failed:`, e)
      }
    }, 'image/jpeg', 0.7)
  }, [drawDetections])

  const startWebcam = useCallback(async () => {
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setIsRunning(true)
      intervalRef.current = setInterval(captureAndSend, FRAME_INTERVAL)
    } catch (e) {
      setError(e.message || 'Could not access webcam. Please allow camera permissions.')
    }
  }, [captureAndSend])

  const stopWebcam = useCallback(() => {
    clearInterval(intervalRef.current)
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) videoRef.current.srcObject = null
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    setIsRunning(false)
    setFps(0)
    setFrameCount(0)
    setResult(null)
  }, [])

  useEffect(() => () => stopWebcam(), [stopWebcam])

  return (
    <div className="webcam-root">
      {/* Stats bar */}
      <div className="webcam-stats glass-card">
        <div className="wstat">
          <span className="wstat-value" style={{ color: isRunning ? '#22c55e' : 'var(--text-muted)' }}>
            {isRunning ? 'LIVE' : 'STOPPED'}
          </span>
          <span className="wstat-label">Status</span>
        </div>
        <div className="wstat-divider" />
        <div className="wstat">
          <span className="wstat-value">{fps}</span>
          <span className="wstat-label">Infer FPS</span>
        </div>
        <div className="wstat-divider" />
        <div className="wstat">
          <span className="wstat-value">{result?.count ?? 0}</span>
          <span className="wstat-label">Detections</span>
        </div>
        <div className="wstat-divider" />
        <div className="wstat">
          <span className="wstat-value">{frameCount}</span>
          <span className="wstat-label">Frames Sent</span>
        </div>
        <div className="wstat-divider" />
        <div className="wstat">
          <span className="wstat-value">
            {result?.inference_ms != null ? `${result.inference_ms.toFixed(0)}ms` : '—'}
          </span>
          <span className="wstat-label">Latency</span>
        </div>

        <div className="webcam-controls">
          {!isRunning ? (
            <button className="btn-start" onClick={startWebcam} id="btn-start-webcam">
              <span className="btn-dot live-dot" />
              Start Camera
            </button>
          ) : (
            <button className="btn-stop" onClick={stopWebcam} id="btn-stop-webcam">
              ■ Stop
            </button>
          )}
        </div>
      </div>

      <div className="webcam-workspace">
        {/* Video + canvas overlay */}
        <div className="video-column">
          <div className="video-container glass-card">
            {!isRunning && !error && (
              <div className="video-placeholder">
                <div className="cam-icon">
                  <svg viewBox="0 0 64 64" fill="none" width="56" height="56">
                    <rect x="4" y="16" width="44" height="32" rx="6" stroke="currentColor" strokeWidth="2"/>
                    <path d="M48 26L60 20V44L48 38V26Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                    <circle cx="26" cy="32" r="8" stroke="currentColor" strokeWidth="2"/>
                    <circle cx="26" cy="32" r="3" fill="currentColor" fillOpacity="0.4"/>
                  </svg>
                </div>
                <p className="placeholder-title">Camera Feed</p>
                <p className="placeholder-sub">Click "Start Camera" to begin live detection</p>
              </div>
            )}

            {error && (
              <div className="video-placeholder error-state">
                <div className="error-cam-icon">⚠️</div>
                <p className="placeholder-title">Camera Error</p>
                <p className="placeholder-sub">{error}</p>
                <button className="btn-retry" onClick={startWebcam}>Try Again</button>
              </div>
            )}

            <video
              ref={videoRef}
              className={`webcam-video ${isRunning ? 'visible' : 'hidden'}`}
              muted
              playsInline
              autoPlay
              id="webcam-video"
            />
            <canvas
              ref={canvasRef}
              className={`detection-canvas ${isRunning ? 'visible' : 'hidden'}`}
              id="detection-canvas"
            />

            {isRunning && (
              <div className="live-badge">
                <span className="live-dot-blink" />
                LIVE
              </div>
            )}
          </div>

          {/* Per-class live counts */}
          {isRunning && result?.by_class && (
            <div className="live-classes glass-card">
              <span className="live-classes-title">Live Detections</span>
              <div className="live-classes-row">
                {Object.entries(result.by_class).map(([cls, cnt]) => {
                  const colour = classColours?.[cls] || '#6366f1'
                  return (
                    <div
                      key={cls}
                      className="live-class-chip"
                      style={{ borderColor: `${colour}55`, color: colour, background: `${colour}18` }}
                    >
                      <span style={{ background: colour }} className="chip-dot" />
                      <span>{cls}</span>
                      <strong>{cnt}</strong>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>

        {/* Side results panel */}
        {result && (
          <div className="webcam-results">
            <ResultPane result={result} classColours={classColours} />
          </div>
        )}
      </div>
    </div>
  )
}
