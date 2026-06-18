import { useState, useEffect, useCallback } from 'react'
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom'
import UploadPane from './components/UploadPane.jsx'
import WebcamPane from './components/WebcamPane.jsx'
import VideoPane from './components/VideoPane.jsx'
import './App.css'

const TABS = [
  { id: 'upload', label: 'Upload Image', icon: '🖼️', path: '/' },
  { id: 'webcam', label: 'Webcam',       icon: '📷', path: '/webcam' },
  { id: 'video',  label: 'Video',        icon: '🎬', path: '/video' },
]

const CLASS_COLOURS = {
  bottle: '#FFC800',
  box: '#64FF64',
  can: '#00A0FF',
  bag: '#C800FF',
}

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()

  // Derive active tab from current URL path
  const activeTab = TABS.find((t) => t.path === location.pathname)?.id ?? 'upload'
  const [health, setHealth] = useState({ status: 'checking', latency: null })

  const checkHealth = useCallback(async () => {
    const t0 = performance.now()
    try {
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
      const res = await fetch(`${API_BASE}/health`)
      const latency = Math.round(performance.now() - t0)
      if (res.ok) {
        const data = await res.json()
        setHealth({ status: data.status || 'ok', latency })
      } else {
        console.error(`HTTP request to /health failed with status ${res.status}`)
        setHealth({ status: 'error', latency: null })
      }
    } catch (e) {
      console.error(`HTTP request to /health failed:`, e)
      setHealth({ status: 'offline', latency: null })
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 10000)
    return () => clearInterval(interval)
  }, [checkHealth])

  const healthColor =
    health.status === 'ok' || health.status === 'healthy'
      ? '#22c55e'
      : health.status === 'checking'
      ? '#f59e0b'
      : '#ef4444'

  return (
    <div className="app-root">
      {/* Animated background orbs */}
      <div className="bg-orb bg-orb-1" />
      <div className="bg-orb bg-orb-2" />
      <div className="bg-orb bg-orb-3" />

      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-logo">
          <div className="logo-icon">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="8" fill="url(#logoGrad)" />
              <rect x="6" y="8" width="8" height="10" rx="2" fill="white" fillOpacity="0.9" />
              <rect x="18" y="8" width="8" height="10" rx="2" fill="white" fillOpacity="0.6" />
              <rect x="6" y="21" width="20" height="3" rx="1.5" fill="white" fillOpacity="0.4" />
              <circle cx="10" cy="27" r="1.5" fill="white" fillOpacity="0.7" />
              <circle cx="22" cy="27" r="1.5" fill="white" fillOpacity="0.7" />
              <defs>
                <linearGradient id="logoGrad" x1="0" y1="0" x2="32" y2="32">
                  <stop stopColor="#6366f1" />
                  <stop offset="1" stopColor="#8b5cf6" />
                </linearGradient>
              </defs>
            </svg>
          </div>
          <div className="logo-text">
            <span className="logo-name">ShelfSight</span>
            <span className="logo-sub">Retail Object Detector</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-label">Navigation</div>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => navigate(tab.path)}
              id={`nav-${tab.id}`}
            >
              <span className="nav-icon">{tab.icon}</span>
              <span className="nav-text">{tab.label}</span>
              {activeTab === tab.id && <div className="nav-indicator" />}
            </button>
          ))}
        </nav>

        {/* Health Status */}
        <div className="sidebar-health glass-card">
          <div className="health-header">
            <span className="health-title">Backend Status</span>
            <button className="health-refresh" onClick={checkHealth} title="Refresh">↻</button>
          </div>
          <div className="health-row">
            <span
              className="health-dot"
              style={{ background: healthColor, boxShadow: `0 0 8px ${healthColor}` }}
            />
            <span className="health-status" style={{ color: healthColor }}>
              {health.status.toUpperCase()}
            </span>
          </div>
          {health.latency !== null && (
            <div className="health-latency">{health.latency}ms latency</div>
          )}
          <div className="health-endpoint">localhost:8000</div>
        </div>

        {/* Class Legend */}
        <div className="sidebar-legend glass-card">
          <div className="legend-title">Detection Classes</div>
          {Object.entries(CLASS_COLOURS).map(([cls, colour]) => (
            <div key={cls} className="legend-item">
              <span className="legend-dot" style={{ background: colour, boxShadow: `0 0 6px ${colour}88` }} />
              <span className="legend-cls">{cls}</span>
              <span className="legend-hex" style={{ color: colour }}>{colour}</span>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <span>ShelfSight v1.0</span>
          <span>AI Vision Platform</span>
        </div>
      </aside>

      {/* Main content */}
      <main className="main-content">
        <header className="main-header glass-card">
          <div className="header-left">
            <h1 className="header-title">
              {TABS.find((t) => t.id === activeTab)?.icon}{' '}
              {TABS.find((t) => t.id === activeTab)?.label}
            </h1>
            <p className="header-sub">
              {activeTab === 'upload' && 'Upload a shelf image to detect retail objects'}
              {activeTab === 'webcam' && 'Live webcam feed with real-time object detection'}
              {activeTab === 'video'  && 'Upload a video file for batch detection analysis'}
            </p>
          </div>
          <div className="header-badge">
            <span className="badge-dot" style={{ background: healthColor }} />
            AI Model Active
          </div>
        </header>

        <div className="pane-container">
          <Routes>
            <Route path="/"       element={<UploadPane classColours={CLASS_COLOURS} />} />
            <Route path="/webcam" element={<WebcamPane classColours={CLASS_COLOURS} />} />
            <Route path="/video"  element={<VideoPane  classColours={CLASS_COLOURS} />} />
            {/* Fallback — redirect unknown paths to upload */}
            <Route path="*"       element={<UploadPane classColours={CLASS_COLOURS} />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
