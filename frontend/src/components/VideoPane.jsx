import { useState, useRef } from 'react'
import './VideoPane.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function VideoPane({ classColours }) {
  const [file, setFile] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [videoUrl, setVideoUrl] = useState(null)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const videoRef = useRef(null)

  const handleFile = (f) => {
    if (!f || !f.type.startsWith('video/')) {
      setError('Please upload a valid video file (MP4, AVI, MOV)')
      return
    }
    setFile(f)
    setVideoUrl(null)
    setError(null)
  }

  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const onInputChange = (e) => handleFile(e.target.files[0])

  const runInference = async () => {
    if (!file) return
    setLoading(true)
    setProgress(0)
    setError(null)
    setVideoUrl(null)

    // Simulate progress while waiting
    const progressInterval = setInterval(() => {
      setProgress((p) => Math.min(p + Math.random() * 8, 92))
    }, 400)

    try {
      const fd = new FormData()
      fd.append('file', file)

      const res = await fetch(`${API_BASE}/infer-video`, { method: 'POST', body: fd })
      clearInterval(progressInterval)

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }))
        console.error(`HTTP request to /infer-video failed with status ${res.status}:`, err)
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setProgress(100)
      setTimeout(() => setVideoUrl(data.video_url), 300)
    } catch (e) {
      console.error(`HTTP request to /infer-video failed:`, e)
      clearInterval(progressInterval)
      setError(e.message || 'Failed to process video')
    } finally {
      setLoading(false)
    }
  }

  const clearAll = () => {
    setFile(null)
    setVideoUrl(null)
    setError(null)
    setProgress(0)
    if (inputRef.current) inputRef.current.value = ''
  }

  const formatSize = (bytes) => {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  return (
    <div className="video-root">
      {!videoUrl ? (
        <div className="video-upload-area">
          {/* Drop zone */}
          <div
            className={`video-drop-zone glass-card ${dragOver ? 'drag-over' : ''}`}
            onDrop={onDrop}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onClick={() => !file && !loading && inputRef.current?.click()}
            id="video-drop-zone"
            role="button"
            tabIndex={0}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/*"
              onChange={onInputChange}
              style={{ display: 'none' }}
              id="video-file-input"
            />

            {file ? (
              <div className="video-file-preview">
                <div className="file-icon">
                  <svg viewBox="0 0 64 64" fill="none" width="52" height="52">
                    <rect x="8" y="4" width="40" height="52" rx="6" stroke="currentColor" strokeWidth="2"/>
                    <path d="M22 27L32 33L22 39V27Z" fill="currentColor" fillOpacity="0.8"/>
                    <rect x="14" y="14" width="16" height="2" rx="1" fill="currentColor" fillOpacity="0.4"/>
                    <rect x="14" y="19" width="22" height="2" rx="1" fill="currentColor" fillOpacity="0.4"/>
                  </svg>
                </div>
                <div className="file-info">
                  <p className="file-name">{file.name}</p>
                  <p className="file-size">{formatSize(file.size)}</p>
                </div>
                <button
                  className="file-change-btn"
                  onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
                >
                  Change File
                </button>
              </div>
            ) : (
              <div className="vdz-idle">
                <div className="vdz-icon-wrap">
                  <svg viewBox="0 0 64 64" fill="none" width="44" height="44">
                    <rect x="2" y="12" width="44" height="32" rx="6" stroke="currentColor" strokeWidth="2"/>
                    <path d="M46 24L62 16V48L46 40V24Z" stroke="currentColor" strokeWidth="2" strokeLinejoin="round"/>
                    <path d="M22 28L30 32L22 36V28Z" fill="currentColor" fillOpacity="0.6"/>
                    <path d="M26 40V32M26 32L22 36M26 32L30 36" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                  </svg>
                </div>
                <p className="vdz-title">Drop your video here</p>
                <p className="vdz-sub">or <span className="vdz-link">browse files</span></p>
                <p className="vdz-formats">MP4 · AVI · MOV · MKV</p>
              </div>
            )}
          </div>

          {error && (
            <div className="error-banner" id="video-error-banner">
              <span>⚠</span>
              <span>{error}</span>
              <button onClick={() => setError(null)} className="error-dismiss">✕</button>
            </div>
          )}

          {/* Run button + progress */}
          {file && (
            <div className="video-action-area glass-card">
              <div className="video-action-info">
                <p className="action-title">Ready to analyse</p>
                <p className="action-sub">
                  The backend will process your video frame-by-frame and return an annotated video.
                </p>
              </div>

              {loading ? (
                <div className="video-progress-wrap">
                  <div className="video-progress-header">
                    <span className="progress-label">Processing video…</span>
                    <span className="progress-pct">{Math.round(progress)}%</span>
                  </div>
                  <div className="progress-track">
                    <div
                      className="progress-fill"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <p className="progress-hint">This may take a while for long videos</p>
                </div>
              ) : (
                <div className="video-btn-row">
                  <button className="btn-run-video" onClick={runInference} id="btn-run-video">
                    <span>▶</span> Run Detection
                  </button>
                  <button className="btn-clear-video" onClick={clearAll} id="btn-clear-video">
                    ✕ Clear
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      ) : (
        /* Result video player */
        <div className="video-result-area">
          <div className="result-header glass-card">
            <div className="result-header-info">
              <span className="result-check">✓</span>
              <div>
                <p className="result-title">Detection Complete</p>
                <p className="result-sub">Annotated video is ready to play</p>
              </div>
            </div>
            <button className="btn-new-video" onClick={clearAll} id="btn-new-video">
              + New Video
            </button>
          </div>

          <div className="video-player-wrap glass-card">
            <video
              ref={videoRef}
              src={videoUrl.startsWith('http') ? videoUrl : `${API_BASE}${videoUrl}`}
              controls
              autoPlay
              className="result-video"
              id="result-video"
            />
          </div>

          <div className="video-download glass-card">
            <span className="download-icon">↓</span>
            <div>
              <p className="download-title">Download Annotated Video</p>
              <p className="download-sub">Save the detection result to your device</p>
            </div>
            <a
              href={videoUrl.startsWith('http') ? videoUrl : `${API_BASE}${videoUrl}`}
              download="shelfsight_annotated.mp4"
              className="btn-download"
              id="btn-download-video"
            >
              Download MP4
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
