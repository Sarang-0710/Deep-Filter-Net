import { useState, useRef, useCallback } from 'react'

const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4', 'audio/x-m4a']
const ALLOWED_EXT   = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

function isAllowedFile(file) {
  if (ALLOWED_TYPES.includes(file.type)) return true
  const name = file.name.toLowerCase()
  return ALLOWED_EXT.some(ext => name.endsWith(ext))
}

export default function App() {
  const [file, setFile]           = useState(null)         // original File object
  const [originalUrl, setOriginalUrl] = useState(null)     // blob URL for playback
  const [processing, setProcessing] = useState(false)
  const [resultUrl, setResultUrl]  = useState(null)         // blob URL for cleaned audio
  const [error, setError]          = useState(null)
  const [dragOver, setDragOver]    = useState(false)

  const fileInputRef = useRef(null)

  // ---- file selection ----
  const selectFile = useCallback((f) => {
    setError(null)
    setResultUrl(null)

    if (!isAllowedFile(f)) {
      setError(`Unsupported file type. Please upload a WAV, MP3, FLAC, OGG, or M4A file.`)
      return
    }

    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setFile(f)
    setOriginalUrl(URL.createObjectURL(f))
  }, [originalUrl])

  const onFileInputChange = (e) => {
    if (e.target.files[0]) selectFile(e.target.files[0])
  }

  // ---- drag-and-drop ----
  const onDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) selectFile(dropped)
  }

  // ---- process ----
  const handleProcess = async () => {
    if (!file) return
    setProcessing(true)
    setError(null)
    if (resultUrl) {
      URL.revokeObjectURL(resultUrl)
      setResultUrl(null)
    }

    try {
      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch('/api/process', {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        let msg = `Server error (${res.status})`
        try {
          const json = await res.json()
          msg = json.detail || msg
        } catch (_) {}
        throw new Error(msg)
      }

      const blob = await res.blob()
      setResultUrl(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div className="page">
      <header className="app-header">
        <h1>DeepFilterNet Audio Denoiser</h1>
        <p className="subtitle">Upload a noisy audio file — the model will clean it up.</p>
      </header>

      <main className="card">
        {/* ── Drop zone ── */}
        <div
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
          onClick={() => fileInputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current.click()}
          id="drop-zone"
          aria-label="Audio file drop zone"
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
            onChange={onFileInputChange}
            style={{ display: 'none' }}
            id="file-input"
          />
          {file ? (
            <>
              <span className="drop-icon">🎵</span>
              <span className="drop-filename">{file.name}</span>
              <span className="drop-hint">Click or drag to replace</span>
            </>
          ) : (
            <>
              <span className="drop-icon">📂</span>
              <span className="drop-label">Click or drag &amp; drop an audio file here</span>
              <span className="drop-hint">WAV · MP3 · FLAC · OGG · M4A</span>
            </>
          )}
        </div>

        {/* ── Original player ── */}
        {originalUrl && (
          <section className="player-section">
            <h2 className="player-label">Original</h2>
            <audio controls src={originalUrl} className="audio-player" id="original-player" />
          </section>
        )}

        {/* ── Error ── */}
        {error && (
          <div className="error-box" role="alert" id="error-message">
            ⚠️ {error}
          </div>
        )}

        {/* ── Process button ── */}
        <button
          className={`process-btn ${processing ? 'loading' : ''}`}
          onClick={handleProcess}
          disabled={!file || processing}
          id="process-button"
        >
          {processing ? (
            <><span className="spinner" aria-hidden="true" /> Processing…</>
          ) : (
            '✨ Clean Audio'
          )}
        </button>

        {/* ── Result player + download ── */}
        {resultUrl && (
          <section className="player-section result-section">
            <h2 className="player-label">Cleaned Result</h2>
            <audio controls src={resultUrl} className="audio-player" id="result-player" autoPlay />
            <a
              href={resultUrl}
              download="denoised.wav"
              className="download-btn"
              id="download-button"
            >
              ⬇ Download denoised.wav
            </a>
          </section>
        )}
      </main>

      <footer className="app-footer">
        Powered by <a href="https://github.com/rikorose/DeepFilterNet" target="_blank" rel="noreferrer">DeepFilterNet3</a>
        &nbsp;·&nbsp; local dev only
      </footer>
    </div>
  )
}
