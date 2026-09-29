import { useState, useRef, useCallback } from 'react'

const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4', 'audio/x-m4a']
const ALLOWED_EXT   = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']

function isAllowedFile(file) {
  if (ALLOWED_TYPES.includes(file.type)) return true
  return ALLOWED_EXT.some(ext => file.name.toLowerCase().endsWith(ext))
}

export default function App() {
  const [file, setFile]             = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [resultUrl, setResultUrl]   = useState(null)
  const [error, setError]           = useState(null)
  const [dragOver, setDragOver]     = useState(false)
  
  // Stopwatch for FULL file processing time
  const [latencyMs, setLatencyMs]   = useState(null)

  const fileInputRef  = useRef(null)

  // ── File selection ────────────────────────────────────────────────────────
  const selectFile = useCallback((f) => {
    setError(null)
    setResultUrl(null)
    setLatencyMs(null)
    if (!isAllowedFile(f)) {
      setError('Unsupported file type. Please upload a WAV, MP3, FLAC, OGG, or M4A file.')
      return
    }
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    setFile(f)
    setOriginalUrl(URL.createObjectURL(f))
  }, [originalUrl])

  const onFileInputChange = (e) => { if (e.target.files[0]) selectFile(e.target.files[0]) }
  const onDrop = (e) => {
    e.preventDefault(); setDragOver(false)
    if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0])
  }

  // ── Full File Processing (Multi-Agent Simulation) ──────────────────────────
  const handleProcess = async () => {
    if (!file || processing) return
    
    setProcessing(true)
    setError(null)
    setLatencyMs(null)
    
    if (resultUrl) { 
        URL.revokeObjectURL(resultUrl); 
        setResultUrl(null) 
    }

    const startTime = performance.now()

    try {
      const formData = new FormData()
      formData.append('file', file)

      // This endpoint uses ParallelOnnxEnhancer under the hood!
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
      
      // Stop the stopwatch the exact moment the FULL file is received
      const endTime = performance.now()
      setLatencyMs(Math.round(endTime - startTime))
      
      setResultUrl(URL.createObjectURL(blob))
    } catch (err) {
      setError(err.message)
    } finally {
      setProcessing(false)
    }
  }

  // ── UI ────────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <header className="app-header">
        <h1>DeepFilterNet (Parallel Agent Mode)</h1>
        <p className="subtitle">This tests the standalone agent. The whole file is processed at once.</p>
      </header>

      <main className="card">
        {/* Drop zone */}
        <div
          className={`drop-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
          onClick={() => fileInputRef.current.click()}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          role="button" tabIndex={0}
          onKeyDown={(e) => e.key === 'Enter' && fileInputRef.current.click()}
          id="drop-zone" aria-label="Audio file drop zone"
        >
          <input
            ref={fileInputRef} type="file"
            accept=".wav,.mp3,.flac,.ogg,.m4a,audio/*"
            onChange={onFileInputChange}
            style={{ display: 'none' }} id="file-input"
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

        {/* Original player */}
        {originalUrl && (
          <section className="player-section">
            <h2 className="player-label">Original</h2>
            <audio controls src={originalUrl} className="audio-player" id="original-player" />
          </section>
        )}

        {/* Error */}
        {error && (
          <div className="error-box" role="alert" id="error-message">⚠️ {error}</div>
        )}

        {/* Process button */}
        <button
          className={`process-btn ${processing ? 'loading' : ''}`}
          onClick={handleProcess}
          disabled={!file || processing}
          id="process-button"
        >
          {processing ? (
            <><span className="spinner" aria-hidden="true" /> Processing Full File…</>
          ) : (
            '✨ Clean Audio (Parallel Mode)'
          )}
        </button>

        {/* Stopwatch display */}
        {latencyMs !== null && (
          <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            ⏱️ Total time for FULL processing: <strong style={{ color: 'var(--accent)' }}>{latencyMs} ms</strong>
          </div>
        )}

        {/* Result player + download */}
        {resultUrl && (
          <section className="player-section result-section">
            <h2 className="player-label">Cleaned Result</h2>
            <audio controls src={resultUrl} className="audio-player" id="result-player" autoPlay />
            <a href={resultUrl} download="denoised.wav" className="download-btn" id="download-button">
              ⬇ Download denoised.wav
            </a>
          </section>
        )}
      </main>

      <footer className="app-footer">
        Powered by <a href="https://github.com/rikorose/DeepFilterNet" target="_blank" rel="noreferrer">DeepFilterNet3</a>
        &nbsp;·&nbsp; Full File Parallel Processing
      </footer>
    </div>
  )
}
