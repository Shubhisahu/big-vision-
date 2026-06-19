import { useState, useRef, useCallback } from 'react'
import ResultPane from './ResultPane.jsx'
import './UploadPane.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function UploadPane({ classColours }) {
  const [dragOver, setDragOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [threshold, setThreshold] = useState(0.35)
  const [previewUrl, setPreviewUrl] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback(async (file) => {
    if (!file) return;

    // Block explicitly wrong files (videos, pdfs), but allow everything else
    // through to the backend (to account for weird Apple OS MIME/extension quirks)
    if (file.type.startsWith('video/') || file.type === 'application/pdf') {
      setError('Please upload a valid image file, not a video or document.')
      return
    }

    const objectUrl = URL.createObjectURL(file)
    setPreviewUrl(objectUrl)
    setResult(null)
    setError(null)
    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)
      formData.append('confidence', threshold)

      const res = await fetch(`${API_BASE}/infer`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Server error' }))
        console.error(`HTTP request to /infer failed with status ${res.status}:`, err)
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
    } catch (e) {
      console.error(`HTTP request to /infer failed:`, e)
      setError(e.message || 'Failed to connect to backend')
    } finally {
      setLoading(false)
    }
  }, [threshold])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  const onDragOver = (e) => { e.preventDefault(); setDragOver(true) }
  const onDragLeave = () => setDragOver(false)

  const onInputChange = (e) => {
    const file = e.target.files[0]
    if (file) handleFile(file)
  }

  const clearAll = () => {
    setResult(null)
    setError(null)
    setPreviewUrl(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div className="upload-root">
      {/* Controls Row */}
      <div className="upload-controls glass-card">
        <div className="threshold-group">
          <label className="ctrl-label" htmlFor="conf-slider">
            Confidence Threshold
            <span className="threshold-value">{threshold.toFixed(2)}</span>
          </label>
          <input
            id="conf-slider"
            type="range"
            min="0.1"
            max="1.0"
            step="0.05"
            value={threshold}
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>0.10</span>
            <span>0.55</span>
            <span>1.00</span>
          </div>
        </div>
        {(result || previewUrl) && (
          <button className="btn-clear" onClick={clearAll} id="btn-clear-upload">
            ✕ Clear
          </button>
        )}
      </div>

      <div className="upload-workspace">
        {/* Left: Upload Zone + Preview */}
        <div className="upload-left">
          <div
            className={`drop-zone glass-card ${dragOver ? 'drag-over' : ''} ${loading ? 'loading' : ''}`}
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onClick={() => !loading && inputRef.current?.click()}
            id="drop-zone"
            role="button"
            tabIndex={0}
            onKeyDown={(e) => e.key === 'Enter' && !loading && inputRef.current?.click()}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/*,.heic,.heif"
              onChange={onInputChange}
              style={{ display: 'none' }}
              id="file-input"
            />

            {loading ? (
              <div className="dz-loading">
                <div className="spinner large-spinner" />
                <p className="dz-loading-text">Running inference…</p>
                <p className="dz-loading-sub">Detecting objects in your image</p>
              </div>
            ) : result?.annotated_image ? (
              <div className="dz-result-image">
                <img
                  src={`data:image/jpeg;base64,${result.annotated_image}`}
                  alt="Annotated detection result"
                  className="annotated-img"
                />
                <div className="img-overlay-badge">
                  <span>✓</span> {result.count} object{result.count !== 1 ? 's' : ''} detected
                </div>
                <div className="reupload-hint">Click or drop to upload new image</div>
              </div>
            ) : previewUrl ? (
              <div className="dz-preview">
                <img src={previewUrl} alt="Preview" className="preview-img" />
                <div className="preview-overlay">Processing…</div>
              </div>
            ) : (
              <div className="dz-idle">
                <div className="dz-icon-wrap">
                  <svg className="dz-icon" viewBox="0 0 64 64" fill="none">
                    <rect x="8" y="16" width="48" height="36" rx="6" stroke="currentColor" strokeWidth="2.5" strokeDasharray="4 3"/>
                    <path d="M32 38V26M32 26L26 32M32 26L38 32" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                    <circle cx="50" cy="14" r="6" fill="url(#uploadGrad)" />
                    <path d="M48 14H52M50 12V16" stroke="white" strokeWidth="1.5" strokeLinecap="round"/>
                    <defs>
                      <linearGradient id="uploadGrad" x1="44" y1="8" x2="56" y2="20">
                        <stop stopColor="#6366f1"/>
                        <stop offset="1" stopColor="#8b5cf6"/>
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
                <p className="dz-title">Drop your image here</p>
                <p className="dz-sub">or <span className="dz-link">browse files</span></p>
                <p className="dz-formats">JPG · PNG · WEBP · HEIC</p>
              </div>
            )}
          </div>

          {error && (
            <div className="error-banner" id="error-banner">
              <span className="error-icon">⚠</span>
              <span>{error}</span>
              <button onClick={() => setError(null)} className="error-dismiss">✕</button>
            </div>
          )}
        </div>

        {/* Right: Results */}
        {result && (
          <div className="upload-right">
            <ResultPane result={result} classColours={classColours} />
          </div>
        )}
      </div>
    </div>
  )
}
