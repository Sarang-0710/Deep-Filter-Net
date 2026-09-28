import { useState, useRef, useCallback, useEffect } from 'react'

const ALLOWED_TYPES = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/ogg', 'audio/mp4', 'audio/x-m4a']
const ALLOWED_EXT   = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
// Derive WebSocket URL from current page origin so it works through Vite proxy
const WS_URL        = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/api/stream`
const TARGET_SR     = 48000
const CHUNK_SEC     = 4.0      // seconds per streaming chunk
const DONE_SENTINEL = 0xFFFFFFFF

function isAllowedFile(file) {
  if (ALLOWED_TYPES.includes(file.type)) return true
  return ALLOWED_EXT.some(ext => file.name.toLowerCase().endsWith(ext))
}

/** Decode an audio File to mono Float32Array at 48 kHz using Web Audio API */
async function decodeAudioFile(file) {
  const ctx    = new AudioContext({ sampleRate: TARGET_SR })
  const buf    = await file.arrayBuffer()
  const decoded = await ctx.decodeAudioData(buf)

  // Mix down to mono
  let mono
  if (decoded.numberOfChannels === 1) {
    mono = decoded.getChannelData(0).slice()
  } else {
    mono = new Float32Array(decoded.length)
    for (let ch = 0; ch < decoded.numberOfChannels; ch++) {
      const chData = decoded.getChannelData(ch)
      for (let i = 0; i < decoded.length; i++) mono[i] += chData[i]
    }
    const inv = 1 / decoded.numberOfChannels
    for (let i = 0; i < mono.length; i++) mono[i] *= inv
  }
  await ctx.close()
  return mono   // Float32Array @ TARGET_SR, mono
}

/** Pack: [uint32 n_samples][float32 * n_samples] */
function encodeFrame(samples) {
  const buf = new ArrayBuffer(4 + samples.length * 4)
  new DataView(buf).setUint32(0, samples.length, true)
  new Float32Array(buf, 4).set(samples)
  return buf
}

/** Send end-of-stream sentinel: [uint32 0xFFFFFFFF] */
function eosFrame() {
  const buf = new ArrayBuffer(4)
  new DataView(buf).setUint32(0, DONE_SENTINEL, true)
  return buf
}

/** Decode a server binary frame. Returns { done: true } or { samples: Float32Array } */
function decodeFrame(data) {
  const view    = new DataView(data)
  const n       = view.getUint32(0, true)
  if (n === DONE_SENTINEL) return { done: true }
  if (n === 0)             return { samples: new Float32Array(0) }
  return { samples: new Float32Array(data, 4, n) }
}

export default function App() {
  const [file, setFile]             = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [resultUrl, setResultUrl]   = useState(null)
  const [error, setError]           = useState(null)
  const [dragOver, setDragOver]     = useState(false)
  const [progress, setProgress]     = useState(0)    // 0–100
  const [statusText, setStatusText] = useState('')
  
  // Stopwatch state
  const [latencyMs, setLatencyMs]       = useState(null)
  const [timerRunning, setTimerRunning] = useState(false)
  
  const startTimeRef    = useRef(null)
  const firstChunkRef   = useRef(false)

  const fileInputRef  = useRef(null)
  const wsRef         = useRef(null)
  const audioCtxRef   = useRef(null)
  const nextPlayTime  = useRef(0)
  const collectedRef  = useRef([])   // accumulated denoised samples for download

  // Cleanup on unmount
  useEffect(() => () => {
    wsRef.current?.close()
    audioCtxRef.current?.close()
  }, [])

  // Timer loop for the stopwatch
  useEffect(() => {
    let frameId
    const update = () => {
      if (timerRunning && startTimeRef.current) {
        setLatencyMs(Math.round(performance.now() - startTimeRef.current))
        frameId = requestAnimationFrame(update)
      }
    }
    if (timerRunning) frameId = requestAnimationFrame(update)
    return () => cancelAnimationFrame(frameId)
  }, [timerRunning])

  // ── File selection ────────────────────────────────────────────────────────
  const selectFile = useCallback((f) => {
    setError(null)
    setResultUrl(null)
    setProgress(0)
    setStatusText('')
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

  // ── Play a denoised chunk via Web Audio API ───────────────────────────────
  const scheduleChunk = useCallback((samples) => {
    if (!audioCtxRef.current || samples.length === 0) return
    const ctx      = audioCtxRef.current
    const audioBuf = ctx.createBuffer(1, samples.length, TARGET_SR)
    audioBuf.getChannelData(0).set(samples)
    const src      = ctx.createBufferSource()
    src.buffer     = audioBuf
    src.connect(ctx.destination)

    const now = ctx.currentTime
    const startAt = Math.max(now, nextPlayTime.current)
    src.start(startAt)
    nextPlayTime.current = startAt + audioBuf.duration
  }, [])

  // ── Build downloadable WAV blob from accumulated PCM ─────────────────────
  const buildWavBlob = useCallback(() => {
    const all    = collectedRef.current
    const total  = all.reduce((s, a) => s + a.length, 0)
    const merged = new Float32Array(total)
    let offset   = 0
    for (const chunk of all) { merged.set(chunk, offset); offset += chunk.length }

    // Minimal WAV header for 48kHz mono float32
    const dataBytes  = merged.buffer.byteLength
    const header     = new ArrayBuffer(44)
    const view       = new DataView(header)
    const writeStr   = (s, o) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)) }
    writeStr('RIFF', 0); view.setUint32(4, 36 + dataBytes, true)
    writeStr('WAVE', 8); writeStr('fmt ', 12)
    view.setUint32(16, 16, true);   view.setUint16(20, 3, true)   // PCM float
    view.setUint16(22, 1, true);    view.setUint32(24, TARGET_SR, true)
    view.setUint32(28, TARGET_SR * 4, true); view.setUint16(32, 4, true); view.setUint16(34, 32, true)
    writeStr('data', 36); view.setUint32(40, dataBytes, true)
    return new Blob([header, merged.buffer], { type: 'audio/wav' })
  }, [])

  // ── Main streaming handler ────────────────────────────────────────────────
  const handleProcess = useCallback(async () => {
    if (!file || processing) return
    setProcessing(true)
    setError(null)
    setProgress(0)
    setStatusText('Decoding audio…')
    setLatencyMs(0)
    setTimerRunning(true)
    startTimeRef.current = performance.now()
    firstChunkRef.current = false
    
    if (resultUrl) { URL.revokeObjectURL(resultUrl); setResultUrl(null) }

    collectedRef.current  = []
    nextPlayTime.current  = 0

    // Create a fresh AudioContext for this session
    if (audioCtxRef.current) await audioCtxRef.current.close()
    audioCtxRef.current = new AudioContext({ sampleRate: TARGET_SR })
    nextPlayTime.current = audioCtxRef.current.currentTime + 0.1

    let pcm
    try {
      pcm = await decodeAudioFile(file)
    } catch (err) {
      setError('Failed to decode audio: ' + err.message)
      setProcessing(false)
      return
    }

    const chunkSize    = Math.round(CHUNK_SEC * TARGET_SR)
    const totalSamples = pcm.length
    let sentSamples    = 0

    setStatusText('Connecting…')

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws
    ws.binaryType = 'arraybuffer'

    let prewarmAcked  = false
    let chunkIndex    = 0
    let prewarmChunk  = null

    // Divide PCM into chunks. First chunk = prewarm context.
    const chunks = []
    for (let i = 0; i < totalSamples; i += chunkSize) {
      chunks.push(pcm.subarray(i, Math.min(i + chunkSize, totalSamples)))
    }
    // First chunk is prewarm (at least 1s = 48000 samples; we use one chunk for simplicity)
    prewarmChunk = chunks[0]
    const dataChunks = chunks.slice(1)

    ws.onopen = () => {
      setStatusText('Processing…')
      // Send prewarm frame
      ws.send(encodeFrame(prewarmChunk))
      sentSamples += prewarmChunk.length
    }

    ws.onmessage = (ev) => {
      const frame = decodeFrame(ev.data)

      if (frame.done) {
        // All chunks processed — build WAV and offer download
        const blob = buildWavBlob()
        const url  = URL.createObjectURL(blob)
        setResultUrl(url)
        setProcessing(false)
        setTimerRunning(false)
        setProgress(100)
        setStatusText('Done!')
        ws.close()
        return
      }

      if (!prewarmAcked) {
        // Server acknowledged prewarm — now send real chunks one by one
        prewarmAcked = true
        sendNextChunk()
        return
      }

      // Received a denoised chunk — play it immediately
      if (frame.samples && frame.samples.length > 0) {
        if (!firstChunkRef.current) {
          firstChunkRef.current = true
          setTimerRunning(false)
          setLatencyMs(Math.round(performance.now() - startTimeRef.current))
        }
        
        const copy = frame.samples.slice()   // detach from buffer
        collectedRef.current.push(copy)
        scheduleChunk(copy)
      }

      sentSamples += (dataChunks[chunkIndex - 1]?.length ?? 0)
      setProgress(Math.round((chunkIndex / dataChunks.length) * 100))

      sendNextChunk()
    }

    function sendNextChunk() {
      if (chunkIndex >= dataChunks.length) {
        ws.send(eosFrame())
        return
      }
      const chunk = dataChunks[chunkIndex++]
      ws.send(encodeFrame(chunk))
    }

    ws.onerror = (e) => {
      setError('WebSocket error — is the backend running?')
      setProcessing(false)
      setTimerRunning(false)
    }

    ws.onclose = () => {
      if (processing) {
        setProcessing(false)
        setTimerRunning(false)
      }
    }

  }, [file, processing, resultUrl, scheduleChunk, buildWavBlob])

  // ── UI ────────────────────────────────────────────────────────────────────
  return (
    <div className="page">
      <header className="app-header">
        <h1>DeepFilterNet Audio Denoiser</h1>
        <p className="subtitle">Upload a noisy audio file — the model will clean it up in real time.</p>
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
            <>
              <span className="spinner" aria-hidden="true" />
              {statusText || 'Processing…'}
              {progress > 0 && progress < 100 && (
                <span className="progress-pct"> {progress}%</span>
              )}
            </>
          ) : (
            '✨ Clean Audio'
          )}
        </button>

        {/* Stopwatch display */}
        {latencyMs !== null && (
          <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '0.9rem', color: 'var(--text-muted)' }}>
            ⏱️ Time to first audio: <strong style={{ color: 'var(--accent)' }}>{latencyMs} ms</strong>
            {timerRunning && <span style={{ opacity: 0.6 }}> (waiting...)</span>}
          </div>
        )}

        {/* Progress bar */}
        {processing && (
          <div className="progress-bar-wrap" aria-hidden="true">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
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
        &nbsp;·&nbsp; Streaming via WebSocket
      </footer>
    </div>
  )
}
