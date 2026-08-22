import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent, ReactNode, SVGProps } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL?.trim() || 'http://localhost:8000').replace(/\/+$/, '')
const API_ADDRESS = API_BASE.replace(/^https?:\/\//, '')
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v']
const DEMO_JOB_IDS = new Set(['job_demo_citypulse', 'job_demo_launchfilm'])
const MESSAGE_CHUNK_SIZE = 4
const MESSAGE_CHUNK_DELAY_MS = 20

type View = 'workspace' | 'library'
type JobStatus = 'queued' | 'processing' | 'completed' | 'failed'
type ReviewStatus = 'pending' | 'accepted' | 'rejected' | 'revised'

type Highlight = {
  highlight_id: string
  start_sec: number
  end_sec: number
  score: number
  highlight_type: string
  description: string
  reason: string
  review_status: ReviewStatus
}

type DetectionResult = {
  schema_version: '1.0'
  job_id: string
  video: { video_id: string; title: string; duration_sec: number }
  highlights: Highlight[]
}

type Job = {
  job_id: string
  status: JobStatus
  original_name: string
  content_type: string
  size_bytes: number
  language: 'zh' | 'en'
  created_at: string
  updated_at: string
  session_expires_at: string | null
  revision: number
  source_url: string
  error_message: string | null
  result: DetectionResult | null
  messages: ChatMessage[]
}

type ChatMessage = {
  message_id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

type RemoteJob = Omit<Job, 'source_url' | 'messages'>

type EditMessageStreamResult = {
  job: RemoteJob
  reply: string
  changed: boolean
}

type EditMessageStreamEvent =
  | { type: 'start' }
  | { type: 'delta'; delta: string }
  | { type: 'complete'; job: RemoteJob; changed: boolean }
  | { type: 'error'; status: number; detail: string }

type UploadState = {
  phase: 'idle' | 'saving' | 'uploading'
  progress: number
  fileName?: string
}

type IconName =
  | 'spark' | 'upload' | 'library' | 'film' | 'check' | 'close' | 'minus'
  | 'square' | 'source' | 'retry' | 'folder' | 'trash' | 'x'

function Icon({ name, size = 18, ...props }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  const paths: Record<IconName, ReactNode> = {
    spark: <path d="m12 2-1.3 5A5.1 5.1 0 0 1 7 10.7L2 12l5 1.3a5.1 5.1 0 0 1 3.7 3.7l1.3 5 1.3-5a5.1 5.1 0 0 1 3.7-3.7l5-1.3-5-1.3A5.1 5.1 0 0 1 13.3 7L12 2Z" />,
    upload: <><path d="M12 16V4m-5 5 5-5 5 5" /><path d="M5 20h14" /></>,
    library: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M8 4v5" /></>,
    film: <><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M7 4v16m10-16v16M3 9h4m10 0h4M3 15h4m10 0h4" /></>,
    check: <path d="m5 12 4 4L19 6" />,
    close: <path d="m7 7 10 10m0-10L7 17" />,
    minus: <path d="M5 12h14" />,
    square: <rect x="7" y="7" width="10" height="10" rx="1" />,
    source: <><circle cx="12" cy="12" r="8" /><path d="m10 8 6 4-6 4V8Z" /></>,
    retry: <><path d="M20 7v5h-5" /><path d="M19 12a7 7 0 1 1-2-5" /></>,
    folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5v-9Z" />,
    trash: <><path d="M4 7h16M9 7V4h6v3m3 0-1 13H7L6 7" /><path d="M10 11v5m4-5v5" /></>,
    x: <path d="m8 8 8 8m0-8-8 8" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name]}</svg>
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
  return response.json() as Promise<T>
}

async function requestEmpty(path: string, init?: RequestInit): Promise<void> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, init)
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
}

async function requestMessageStream(
  path: string,
  payload: Record<string, unknown>,
  onDelta: (reply: string) => void,
): Promise<EditMessageStreamResult> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)
  }
  if (!response.ok) {
    const error = await response.json().catch(() => null) as { detail?: string } | null
    if (response.status === 404 && error?.detail === 'Not Found' && path.endsWith('/stream')) {
      const fallback = await requestJson<EditMessageStreamResult>(path.slice(0, -'/stream'.length), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      let visibleReply = ''
      for (let index = 0; index < fallback.reply.length; index += MESSAGE_CHUNK_SIZE) {
        visibleReply += fallback.reply.slice(index, index + MESSAGE_CHUNK_SIZE)
        onDelta(visibleReply)
        await new Promise<void>((resolve) => window.setTimeout(resolve, MESSAGE_CHUNK_DELAY_MS))
      }
      return fallback
    }
    throw new Error(error?.detail || `请求失败（${response.status}）`)
  }
  if (!response.body) throw new Error('后端没有返回可读取的流式响应')

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let reply = ''
  let completed: Extract<EditMessageStreamEvent, { type: 'complete' }> | null = null

  const consume = (
    line: string,
  ): Extract<EditMessageStreamEvent, { type: 'complete' }> | null => {
    if (!line.trim()) return null
    let event: EditMessageStreamEvent
    try {
      event = JSON.parse(line) as EditMessageStreamEvent
    } catch {
      throw new Error('后端返回了无效的流式消息')
    }
    if (event.type === 'delta') {
      reply += event.delta
      onDelta(reply)
    } else if (event.type === 'complete') {
      return event
    } else if (event.type === 'error') {
      throw new Error(event.detail)
    }
    return null
  }

  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value, { stream: !done })
    const lines = buffer.split('\n')
    buffer = lines.pop() || ''
    for (const line of lines) completed = consume(line) || completed
    if (done) break
  }
  completed = consume(buffer) || completed
  if (!completed) throw new Error('流式响应在完成前意外结束')
  return { job: completed.job, changed: completed.changed, reply }
}

function uploadVideo(
  jobId: string,
  file: File,
  onProgress: (value: number) => void,
  register: (request: XMLHttpRequest | null) => void,
): Promise<RemoteJob> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    const form = new FormData()
    form.append('job_id', jobId)
    form.append('language', 'zh')
    form.append('file', file, file.name)
    register(request)
    request.open('POST', `${API_BASE}/api/jobs`)
    request.responseType = 'json'
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100))
    }
    request.onload = () => {
      register(null)
      if (request.status >= 200 && request.status < 300) resolve(request.response as RemoteJob)
      else reject(new Error(request.response?.detail || `上传失败（${request.status}）`))
    }
    request.onerror = () => { register(null); reject(new Error(`无法连接后端服务（${API_ADDRESS}），请检查 API 地址、网络和后端状态`)) }
    request.onabort = () => { register(null); reject(new Error('上传已取消')) }
    request.send(form)
  })
}

function mergeRemoteJob(local: Job, remote: RemoteJob): Job {
  const currentHighlights = new Map(
    (local.result?.highlights || []).map((item) => [item.highlight_id, item]),
  )
  const result = remote.result ? {
    ...remote.result,
    highlights: remote.result.highlights.map((item) => {
      const current = currentHighlights.get(item.highlight_id)
      const agentChanged = current && (
        current.start_sec !== item.start_sec
        || current.end_sec !== item.end_sec
        || current.description !== item.description
        || current.reason !== item.reason
      )
      return {
        ...item,
        review_status: agentChanged ? item.review_status : current?.review_status || item.review_status,
      }
    }),
  } : null
  return { ...local, ...remote, result, source_url: local.source_url, messages: local.messages }
}

function createJobId() {
  return `job_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`
}

function hasActiveSession(job: Job) {
  return Boolean(job.session_expires_at && new Date(job.session_expires_at).getTime() > Date.now())
}

function formatBytes(bytes: number) {
  if (!bytes) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  return `${(bytes / 1024 ** index).toFixed(index > 1 ? 1 : 0)} ${units[index]}`
}

function formatTime(seconds: number) {
  const value = Math.max(0, Math.round(seconds))
  return `${String(Math.floor(value / 60)).padStart(2, '0')}:${String(value % 60).padStart(2, '0')}`
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

const statusCopy: Record<JobStatus, string> = {
  queued: '等待分析', processing: '正在提取', completed: '提取完成', failed: '提取失败',
}

function WindowChrome() {
  return <div className="window-chrome">
    <div className="chrome-label"><span /> FRAME / 本地高光工作台</div>
    <div className="window-actions">
      <button aria-label="最小化" onClick={() => window.desktopWindow?.minimize()}><Icon name="minus" size={15} /></button>
      <button aria-label="最大化" onClick={() => window.desktopWindow?.toggleMaximize()}><Icon name="square" size={14} /></button>
      <button className="danger" aria-label="关闭" onClick={() => window.desktopWindow?.close()}><Icon name="close" size={15} /></button>
    </div>
  </div>
}

function Sidebar({ view, onView, online, jobs }: { view: View; onView: (view: View) => void; online: boolean; jobs: Job[] }) {
  const completed = jobs.filter((job) => job.status === 'completed').length
  return <aside className="sidebar">
    <div className="brand"><div className="brand-glyph"><i /><i /><i /></div><div><strong>FRAME</strong><span>高光工作台</span></div></div>
    <nav>
      <small>工作区</small>
      <button className={view === 'workspace' ? 'active' : ''} onClick={() => onView('workspace')}><Icon name="spark" /><span>高光提取</span><b>01</b></button>
      <button className={view === 'library' ? 'active' : ''} onClick={() => onView('library')}><Icon name="library" /><span>任务归档</span><b>02</b></button>
    </nav>
    <div className="sidebar-metric"><span>本地任务</span><strong>{String(jobs.length).padStart(2, '0')}</strong><small>{completed} 个已完成</small></div>
    <div className={`service-state ${online ? 'online' : ''}`}><i /><div><b>{online ? '后端服务在线' : '后端服务离线'}</b><span>API · {API_ADDRESS}</span></div></div>
  </aside>
}

function UploadDropzone({ onFile, compact = false }: { onFile: (file: File) => void; compact?: boolean }) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const receive = (files: FileList | null) => { if (files?.[0]) onFile(files[0]) }
  const drop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    setDragging(false)
    receive(event.dataTransfer.files)
  }
  if (compact) return <>
    <button className="button primary" onClick={() => inputRef.current?.click()}><Icon name="upload" size={17} />导入新视频</button>
    <input ref={inputRef} hidden type="file" accept="video/*,.mkv,.m4v" onChange={(event) => { receive(event.target.files); event.target.value = '' }} />
  </>
  return <div className={`dropzone ${dragging ? 'dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={drop}>
    <input ref={inputRef} hidden type="file" accept="video/*,.mkv,.m4v" onChange={(event) => { receive(event.target.files); event.target.value = '' }} />
    <div className="drop-icon"><Icon name="upload" size={27} /></div>
    <p className="overline">NEW SOURCE / 01</p>
    <h2>把原片放到这里</h2>
    <p>原片与任务记录永久保存在本机；后端只临时接收视频并返回高光时间段。</p>
    <button className="button primary" onClick={() => inputRef.current?.click()}><Icon name="film" size={17} />选择视频文件</button>
    <span>MP4 · MOV · MKV · WEBM · AVI · 最大 20 GB</span>
  </div>
}

function ProgressPanel({ upload, job, onCancel }: { upload: UploadState; job: Job | null; onCancel: () => void }) {
  const uploading = upload.phase !== 'idle'
  const label = upload.phase === 'saving' ? '正在保存本地原片' : upload.phase === 'uploading' ? `正在上传 ${upload.progress}%` : job ? statusCopy[job.status] : ''
  const number = upload.phase === 'uploading' ? upload.progress : job?.status === 'completed' ? 100 : null
  return <section className={`progress-panel ${job?.status || upload.phase}`}>
    <div className="progress-heading"><div><span className="pulse" /><div><b>{label}</b><small>{uploading ? upload.fileName : job?.original_name}</small></div></div>{upload.phase === 'uploading' && <button onClick={onCancel}><Icon name="x" size={15} />取消上传</button>}</div>
    <div className={`progress-track ${job?.status === 'processing' || job?.status === 'queued' || upload.phase === 'saving' ? 'indeterminate' : ''}`}><i style={{ width: `${number ?? 38}%` }} /></div>
    <div className="phase-rail">
      <span className={upload.phase === 'saving' ? 'current' : upload.phase === 'uploading' || job ? 'done' : ''}><i>1</i>本地保存</span>
      <span className={upload.phase === 'uploading' ? 'current' : job ? 'done' : ''}><i>2</i>上传任务</span>
      <span className={job?.status === 'processing' ? 'current' : job?.status === 'completed' ? 'done' : ''}><i>3</i>Agent 分析</span>
      <span className={job?.status === 'completed' ? 'done' : ''}><i>4</i>等待复核</span>
    </div>
  </section>
}

function Player({ job, selected, onSelect }: { job: Job; selected: Highlight | null; onSelect: (item: Highlight) => void }) {
  const source = job.source_url
  const highlights = job.result?.highlights || []
  const sourceDuration = job.result?.video.duration_sec || 1
  const videoRef = useRef<HTMLVideoElement>(null)
  const [inspectedId, setInspectedId] = useState<string | null>(null)
  const inspectedHighlight = highlights.find((item) => item.highlight_id === inspectedId)
    || selected
    || highlights[0]
    || null
  const inspectedIndex = inspectedHighlight
    ? highlights.findIndex((item) => item.highlight_id === inspectedHighlight.highlight_id)
    : -1
  useEffect(() => {
    const player = videoRef.current
    if (!player) return
    player.src = source
    player.load()
    return () => {
      player.pause()
      player.removeAttribute('src')
      player.load()
    }
  }, [source])
  useEffect(() => {
    const player = videoRef.current
    if (!player) return
    const seekToSelection = () => {
      player.currentTime = selected?.start_sec || 0
      if (selected) void player.play().catch(() => undefined)
    }
    if (player.readyState >= 1) seekToSelection()
    else player.addEventListener('loadedmetadata', seekToSelection, { once: true })
    const stopAtOutPoint = () => {
      if (selected && player.currentTime >= selected.end_sec) {
        player.pause()
        player.currentTime = selected.end_sec
      }
    }
    player.addEventListener('timeupdate', stopAtOutPoint)
    return () => {
      player.removeEventListener('loadedmetadata', seekToSelection)
      player.removeEventListener('timeupdate', stopAtOutPoint)
    }
  }, [selected?.highlight_id, selected?.start_sec, selected?.end_sec])
  return <section className="player-card">
    <div className="player-top"><div><span>VIDEO MONITOR · {selected ? 'HIGHLIGHT' : 'SOURCE'}</span><b>{selected?.description || job.original_name}</b></div><span className="media-chip">{selected ? `${formatTime(selected.start_sec)} — ${formatTime(selected.end_sec)}` : formatBytes(job.size_bytes)}</span></div>
    <div className="video-frame">
      <div className="video-stage"><video ref={videoRef} key={source} controls preload="metadata" /></div>
      <div className="monitor-timeline" aria-label="完整原片高光时间轴">
        <div className="timeline-caption">
          <div><span>完整原片 · 高光轨道</span><small>点击荧光绿区间预览片段</small></div>
          {inspectedHighlight ? <div className="timeline-time-readout" aria-label={`片段 ${inspectedIndex + 1} 的入点和出点`}>
            <small>{`片段 ${String(inspectedIndex + 1).padStart(2, '0')}`}</small>
            <span><i>IN</i><b>{formatTime(inspectedHighlight.start_sec)}</b></span>
            <span><i>OUT</i><b>{formatTime(inspectedHighlight.end_sec)}</b></span>
          </div> : <b>{formatTime(sourceDuration)}</b>}
        </div>
        <div className="timeline-track" role="group" aria-label="可选择的高光片段">
          {highlights.map((item) => {
            const start = Math.max(0, Math.min(100, (item.start_sec / sourceDuration) * 100))
            const end = Math.max(start, Math.min(100, (item.end_sec / sourceDuration) * 100))
            const active = selected?.highlight_id === item.highlight_id
            const inspected = inspectedHighlight?.highlight_id === item.highlight_id
            return <button
              type="button"
              key={item.highlight_id}
              className={`timeline-highlight${active ? ' active' : ''}${inspected ? ' inspected' : ''}`}
              style={{ left: `${start}%`, width: `${end - start}%` }}
              title={`点击预览：${item.description} · 入点 ${formatTime(item.start_sec)} · 出点 ${formatTime(item.end_sec)}`}
              aria-label={`${item.description}，入点 ${formatTime(item.start_sec)}，出点 ${formatTime(item.end_sec)}，点击预览`}
              aria-pressed={active}
              onClick={() => onSelect(item)}
              onMouseEnter={() => setInspectedId(item.highlight_id)}
              onMouseLeave={() => setInspectedId(null)}
              onFocus={() => setInspectedId(item.highlight_id)}
              onBlur={() => setInspectedId(null)}
            />
          })}
        </div>
        <div className="timeline-scale"><span>00:00</span><span>悬停查看出入点 · 点击选择</span><span>{formatTime(sourceDuration)}</span></div>
      </div>
    </div>
  </section>
}

function HighlightList({ job, selectedId, onSelect, onReview }: { job: Job; selectedId: string | null; onSelect: (item: Highlight | null) => void; onReview: (item: Highlight, status: ReviewStatus) => void }) {
  const highlights = job.result?.highlights || []
  return <aside className="result-panel">
    <div className="result-head"><div><p className="overline">AGENT OUTPUT</p><h2>高光候选</h2></div><span>{String(highlights.length).padStart(2, '0')}</span></div>
    <button className={!selectedId ? 'source-row active' : 'source-row'} onClick={() => onSelect(null)}><span><Icon name="source" size={16} /></span><div><b>完整原片</b><small>返回源视频预览</small></div></button>
    <div className="highlight-scroll">
      {highlights.map((item, index) => <article key={item.highlight_id} className={selectedId === item.highlight_id ? 'highlight-card active' : 'highlight-card'}>
        <button className="highlight-main" onClick={() => onSelect(item)}>
          <span className="clip-index">{String(index + 1).padStart(2, '0')}</span>
          <div><div className="clip-meta"><span>{item.highlight_type}</span><b>{Math.round(item.score * 100)}</b></div><h3>{item.description || '未命名高光'}</h3><p>{item.reason}</p><small>{formatTime(item.start_sec)} — {formatTime(item.end_sec)} · {Math.round(item.end_sec - item.start_sec)} 秒</small></div>
        </button>
        <div className="review-actions"><button className={item.review_status === 'accepted' ? 'accepted' : ''} onClick={() => onReview(item, item.review_status === 'accepted' ? 'pending' : 'accepted')}><Icon name="check" size={13} />{item.review_status === 'accepted' ? '已采用' : '采用'}</button><button className={item.review_status === 'rejected' ? 'rejected' : ''} onClick={() => onReview(item, item.review_status === 'rejected' ? 'pending' : 'rejected')}><Icon name="x" size={13} />{item.review_status === 'rejected' ? '已排除' : '排除'}</button></div>
      </article>)}
      {!highlights.length && <div className="empty-results"><Icon name="spark" size={22} /><b>没有生成高光片段</b><p>原片仍已安全保存在任务目录中。</p></div>}
    </div>
  </aside>
}

function HighlightAssistant({ job, selected, busy, onSend }: { job: Job; selected: Highlight | null; busy: boolean; onSend: (message: string) => Promise<void> }) {
  const focus = selected ? `「${selected.description}」` : '完整原片'
  const [input, setInput] = useState('')
  const [now, setNow] = useState(Date.now())
  const threadRef = useRef<HTMLDivElement>(null)
  const latestMessage = job.messages[job.messages.length - 1]
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [])
  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      if (threadRef.current) threadRef.current.scrollTop = threadRef.current.scrollHeight
    })
    return () => window.cancelAnimationFrame(frame)
  }, [job.messages.length, latestMessage?.content])
  const expiresAt = job.session_expires_at ? new Date(job.session_expires_at).getTime() : 0
  const sessionActive = expiresAt > now
  const remainingMinutes = Math.max(0, Math.ceil((expiresAt - now) / 60_000))
  const submit = async (value: string) => {
    const message = value.trim()
    if (!message || busy || !sessionActive) return
    setInput('')
    try {
      await onSend(message)
    } catch {
      setInput(message)
    }
  }
  return <aside className="assistant-panel">
    <div className="assistant-head">
      <div><p className="overline">AI EDIT ASSISTANT</p><h2>高光编辑助手</h2></div>
      <span className={`assistant-state ${sessionActive ? '' : 'expired'}`}><i />{busy ? '处理中' : sessionActive ? `${remainingMinutes} 分钟` : '会话结束'}</span>
    </div>
    <div ref={threadRef} className="assistant-thread" role="log" aria-label="高光编辑助手对话">
      <div className="assistant-context"><span>当前上下文</span><b>{focus}</b><small>{job.result?.highlights.length || 0} 个高光候选已载入</small></div>
      <div className="message assistant-message">
        <span className="message-avatar"><Icon name="spark" size={15} /></span>
        <div><small>EDIT AGENT</small><p>{sessionActive ? `我正在查看${focus}。你可以查询理由，调整时间范围，修改标题或说明，也可以删除、拆分、合并和撤销。` : '服务端临时会话已经结束。本地原片、结果和复核记录不受影响，仍可继续审阅。'}</p></div>
      </div>
      {job.messages.map((message, index) => message.role === 'user'
        ? <div className="message user-message" key={message.message_id}><div><small>YOU</small><p>{message.content}</p></div></div>
        : <div className={`message assistant-message${busy && index === job.messages.length - 1 ? ' streaming' : ''}`} key={message.message_id}><span className="message-avatar"><Icon name="spark" size={15} /></span><div><small>EDIT AGENT</small><p>{message.content}</p></div></div>)}
      <div className="assistant-suggestions"><span>快捷指令</span><div><button type="button" disabled={!selected || !sessionActive || busy} onClick={() => void submit('入点后移 1 秒')}>入点后移 1 秒</button><button type="button" disabled={!selected || !sessionActive || busy} onClick={() => void submit('出点前移 1 秒')}>出点前移 1 秒</button><button type="button" disabled={!sessionActive || busy} onClick={() => void submit('撤销')}>撤销</button></div></div>
    </div>
    <form className="assistant-composer" aria-label="AI 编辑指令输入" onSubmit={(event) => { event.preventDefault(); void submit(input) }}>
      <input value={input} onChange={(event) => setInput(event.target.value)} placeholder={sessionActive ? '询问高光，或描述你想进行的调整…' : '编辑会话已结束'} disabled={!sessionActive || busy} />
      <button type="submit" aria-label="发送编辑指令" disabled={!input.trim() || !sessionActive || busy}>→</button>
    </form>
  </aside>
}

function Workspace({ upload, job, selected, error, chatBusy, onFile, onSelect, onReview, onChat, onCancel, onRetry }: { upload: UploadState; job: Job | null; selected: Highlight | null; error: string | null; chatBusy: boolean; onFile: (file: File) => void; onSelect: (item: Highlight | null) => void; onReview: (item: Highlight, status: ReviewStatus) => void; onChat: (message: string) => Promise<void>; onCancel: () => void; onRetry: () => void }) {
  const busy = upload.phase !== 'idle' || job?.status === 'queued' || job?.status === 'processing'
  const reviewing = job?.status === 'completed'
  return <div className={`view workspace-view${reviewing ? ' task-review' : ''}`}>
    {!reviewing && <header className="view-header"><div><p className="overline">LOCAL VIDEO INTELLIGENCE</p><h1>高光提取</h1><p>导入视频，后台完成保存与高光提取，再由你做最后判断。</p></div><UploadDropzone compact onFile={onFile} /></header>}
    {error && <div className="error-banner"><span><Icon name="close" size={16} /></span><div><b>流程没有完成</b><p>{error}</p></div><button onClick={onRetry}><Icon name="retry" size={15} />重新载入</button></div>}
    {!job && upload.phase === 'idle' && <UploadDropzone onFile={onFile} />}
    {busy && <ProgressPanel upload={upload} job={job} onCancel={onCancel} />}
    {job?.status === 'failed' && <div className="failed-state"><Icon name="close" size={22} /><div><h2>Agent 未能完成分析</h2><p>{job.error_message || '原视频已经安全保存在本机，可以稍后重新提交。'}</p></div></div>}
    {job?.status === 'completed' && <div className="review-grid"><HighlightList job={job} selectedId={selected?.highlight_id || null} onSelect={onSelect} onReview={onReview} /><Player job={job} selected={selected} onSelect={onSelect} /><HighlightAssistant job={job} selected={selected} busy={chatBusy} onSend={onChat} /></div>}
  </div>
}

function Library({ jobs, onOpen, onFile, onDelete }: { jobs: Job[]; onOpen: (job: Job) => void; onFile: (file: File) => void; onDelete: (job: Job) => void }) {
  return <div className="view library-view">
    <header className="view-header"><div><p className="overline">LOCAL ARCHIVE</p><h1>任务归档</h1><p>原片、时间段结果、复核状态和编辑对话永久保存在本机。</p></div><UploadDropzone compact onFile={onFile} /></header>
    <section className="archive-summary"><div><small>全部任务</small><b>{jobs.length}</b></div><div><small>提取完成</small><b>{jobs.filter((item) => item.status === 'completed').length}</b></div><div><small>处理中</small><b>{jobs.filter((item) => ['queued', 'processing'].includes(item.status)).length}</b></div><div><small>高光片段</small><b>{jobs.reduce((sum, item) => sum + (item.result?.highlights.length || 0), 0)}</b></div></section>
    <section className="job-list">
      <div className="job-list-head"><span>视频 / 任务</span><span>大小</span><span>状态</span><span>创建时间</span><span>操作</span></div>
      {jobs.map((item) => {
        const active = item.status === 'queued' || item.status === 'processing'
        return <div className="job-row" key={item.job_id}>
          <button className="job-title" onClick={() => onOpen(item)} aria-label={`打开任务 ${item.original_name}`}><i><Icon name="film" size={17} /></i><span><b>{item.original_name}</b><small>{item.job_id}</small></span></button>
          <span>{formatBytes(item.size_bytes)}</span>
          <span><i className={`status-dot ${item.status}`} />{statusCopy[item.status]}</span>
          <span>{formatDate(item.created_at)}</span>
          <div className="job-actions"><button className="open-job" onClick={() => onOpen(item)} aria-label={`打开任务 ${item.original_name}`}>→</button><button className="delete-job" disabled={active} title={active ? '分析中的任务不能删除' : '删除任务'} onClick={() => onDelete(item)} aria-label={`删除任务 ${item.original_name}`}><Icon name="trash" size={15} /></button></div>
        </div>
      })}
      {!jobs.length && <div className="empty-archive"><Icon name="folder" size={25} /><h2>还没有导入记录</h2><p>从第一条视频开始建立高光任务库。</p></div>}
    </section>
  </div>
}

function DeleteDialog({ job, busy, error, onCancel, onConfirm }: { job: Job; busy: boolean; error: string | null; onCancel: () => void; onConfirm: () => void }) {
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel() }}>
    <section className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
      <div className="dialog-mark"><Icon name="trash" size={21} /></div>
      <p className="overline">DELETE LOCAL TASK</p>
      <h2 id="delete-title">删除这个任务？</h2>
      <p>本地任务记录、原视频、复核状态和编辑对话都会永久删除，此操作无法撤销。</p>
      <div className="delete-target"><Icon name="film" size={17} /><div><b>{job.original_name}</b><small>{job.job_id} · {formatBytes(job.size_bytes)}</small></div></div>
      {error && <div className="dialog-error">{error}</div>}
      <div className="dialog-actions"><button disabled={busy} onClick={onCancel}>取消</button><button className="confirm-delete" disabled={busy} onClick={onConfirm}>{busy ? '正在删除…' : '永久删除'}</button></div>
    </section>
  </div>
}

export default function App() {
  const [view, setView] = useState<View>('workspace')
  const [jobs, setJobs] = useState<Job[]>([])
  const [job, setJob] = useState<Job | null>(null)
  const [selected, setSelected] = useState<Highlight | null>(null)
  const [upload, setUpload] = useState<UploadState>({ phase: 'idle', progress: 0 })
  const [error, setError] = useState<string | null>(null)
  const [online, setOnline] = useState(false)
  const [deleteCandidate, setDeleteCandidate] = useState<Job | null>(null)
  const [deleteBusy, setDeleteBusy] = useState(false)
  const [deleteError, setDeleteError] = useState<string | null>(null)
  const [chatBusy, setChatBusy] = useState(false)
  const uploadRequest = useRef<XMLHttpRequest | null>(null)

  const refreshJobs = useCallback(async () => {
    if (!window.localLibrary) {
      setError('本地任务库只能在 Electron 桌面端使用')
      return []
    }
    try {
      const records = await window.localLibrary.listJobs<Job>()
      setJobs(records)
      try {
        await requestJson<{ status: string }>('/health')
        setOnline(true)
      } catch {
        setOnline(false)
      }
      return records
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '本地任务库读取失败')
      return []
    }
  }, [])

  const persistJob = useCallback(async (record: Job) => {
    if (!window.localLibrary) throw new Error('本地任务库不可用')
    const saved = await window.localLibrary.saveJob(record)
    setJob(saved)
    setJobs((current) => {
      const exists = current.some((item) => item.job_id === saved.job_id)
      const next = exists
        ? current.map((item) => item.job_id === saved.job_id ? saved : item)
        : [saved, ...current]
      return next.sort((left, right) => right.created_at.localeCompare(left.created_at))
    })
    return saved
  }, [])

  useEffect(() => { void refreshJobs() }, [refreshJobs])

  useEffect(() => {
    if (!job || !['queued', 'processing'].includes(job.status)) return
    let cancelled = false
    const timer = window.setInterval(async () => {
      try {
        const latest = await requestJson<RemoteJob>(`/api/jobs/${job.job_id}`)
        if (cancelled) return
        const saved = await persistJob(mergeRemoteJob(job, latest))
        if (cancelled) return
        setOnline(true)
        if (!['queued', 'processing'].includes(saved.status)) {
          window.clearInterval(timer)
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '任务状态查询失败')
      }
    }, 1800)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [job?.job_id, job?.status, persistJob])

  useEffect(() => {
    if (!job?.result?.highlights.length) setSelected(null)
    else if (selected) setSelected(job.result.highlights.find((item) => item.highlight_id === selected.highlight_id) || null)
  }, [job?.updated_at])

  const handleFile = async (file: File) => {
    const extension = file.name.includes('.') ? file.name.slice(file.name.lastIndexOf('.')).toLowerCase() : ''
    if (!VIDEO_EXTENSIONS.includes(extension)) { setError('请选择 MP4、MOV、MKV、WEBM、AVI 或 M4V 视频。'); return }
    if (file.size > 20 * 1024 ** 3) { setError('视频不能超过 20 GB。'); return }
    if (!window.localLibrary) { setError('请在 Electron 桌面端导入视频。'); return }
    const jobId = createJobId()
    let localJob: Job | null = null
    setView('workspace')
    setJob(null)
    setSelected(null)
    setError(null)
    setUpload({ phase: 'saving', progress: 0, fileName: file.name })
    try {
      localJob = await window.localLibrary.importSource<Job>(file, {
        jobId,
        originalName: file.name,
        contentType: file.type,
        language: 'zh',
      })
      setJob(localJob)
      setJobs((current) => [localJob as Job, ...current])
      setUpload({ phase: 'uploading', progress: 0, fileName: file.name })
      const created = await uploadVideo(jobId, file, (progress) => setUpload({ phase: 'uploading', progress, fileName: file.name }), (request) => { uploadRequest.current = request })
      await persistJob(mergeRemoteJob(localJob, created))
      setUpload({ phase: 'idle', progress: 0 })
      setOnline(true)
    } catch (reason) {
      setUpload({ phase: 'idle', progress: 0 })
      const message = reason instanceof Error ? reason.message : '视频导入失败'
      setError(message)
      if (localJob) {
        await persistJob({
          ...localJob,
          status: 'failed',
          updated_at: new Date().toISOString(),
          error_message: `上传或分析任务创建失败：${message}`,
        }).catch(() => undefined)
      }
    }
  }

  const review = async (item: Highlight, reviewStatus: ReviewStatus) => {
    if (!job?.result) return
    try {
      const updated: Job = {
        ...job,
        updated_at: new Date().toISOString(),
        result: {
          ...job.result,
          highlights: job.result.highlights.map((highlight) => highlight.highlight_id === item.highlight_id ? { ...highlight, review_status: reviewStatus } : highlight),
        },
      }
      await persistJob(updated)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '复核状态保存失败')
    }
  }

  const sendEditMessage = async (message: string) => {
    if (!job) return
    const currentJob = job
    setChatBusy(true)
    setError(null)
    const userMessage: ChatMessage = {
      message_id: `msg_${crypto.randomUUID()}`,
      role: 'user',
      content: message,
      created_at: new Date().toISOString(),
    }
    const assistantMessage: ChatMessage = {
      message_id: `msg_${crypto.randomUUID()}`,
      role: 'assistant',
      content: '',
      created_at: new Date().toISOString(),
    }
    setJob({
      ...currentJob,
      messages: [...currentJob.messages, userMessage, assistantMessage],
    })
    try {
      const response = await requestMessageStream(
        `/api/jobs/${currentJob.job_id}/messages/stream`,
        {
          message,
          revision: currentJob.revision,
          selected_highlight_id: selected?.highlight_id || null,
        },
        (reply) => setJob((active) => active?.job_id === currentJob.job_id ? {
          ...active,
          messages: active.messages.map((item) => item.message_id === assistantMessage.message_id
            ? { ...item, content: reply }
            : item),
        } : active),
      )
      const merged = mergeRemoteJob(currentJob, response.job)
      await persistJob({
        ...merged,
        messages: [
          ...currentJob.messages,
          userMessage,
          { ...assistantMessage, content: response.reply },
        ],
      })
      setOnline(true)
    } catch (reason) {
      setJob((active) => active?.job_id === currentJob.job_id ? currentJob : active)
      setError(reason instanceof Error ? reason.message : '高光编辑失败')
      throw reason
    } finally {
      setChatBusy(false)
    }
  }

  const openJob = (record: Job) => {
    setJob(record)
    setSelected(null)
    setError(null)
    setView('workspace')
    if (!DEMO_JOB_IDS.has(record.job_id) || hasActiveSession(record) || !record.result) return

    setChatBusy(true)
    void requestJson<RemoteJob>(`/api/demo-jobs/${record.job_id}/session`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        original_name: record.original_name,
        size_bytes: record.size_bytes,
        language: record.language,
        result: record.result,
      }),
    }).then(async (remote) => {
      await persistJob(mergeRemoteJob(record, remote))
      setOnline(true)
    }).catch((reason) => {
      setError(reason instanceof Error ? reason.message : '演示对话会话创建失败')
    }).finally(() => setChatBusy(false))
  }
  const requestDelete = (record: Job) => { setDeleteCandidate(record); setDeleteError(null) }
  const closeDelete = () => { if (!deleteBusy) { setDeleteCandidate(null); setDeleteError(null) } }
  const confirmDelete = async () => {
    if (!deleteCandidate) return
    setDeleteBusy(true)
    setDeleteError(null)
    try {
      await requestEmpty(`/api/jobs/${deleteCandidate.job_id}`, { method: 'DELETE' }).catch(() => undefined)
      if (!window.localLibrary) throw new Error('本地任务库不可用')
      await window.localLibrary.deleteJob(deleteCandidate.job_id)
      setJobs((current) => current.filter((record) => record.job_id !== deleteCandidate.job_id))
      if (job?.job_id === deleteCandidate.job_id) { setJob(null); setSelected(null) }
      setDeleteCandidate(null)
    } catch (reason) {
      setDeleteError(reason instanceof Error ? reason.message : '任务删除失败')
    } finally {
      setDeleteBusy(false)
    }
  }
  const activeHighlight = useMemo(() => selected && job?.result?.highlights.find((item) => item.highlight_id === selected.highlight_id) || null, [job, selected])

  return <div className="app-shell">
    <WindowChrome />
    <Sidebar view={view} onView={setView} online={online} jobs={jobs} />
    <main>{view === 'workspace'
      ? <Workspace upload={upload} job={job} selected={activeHighlight} error={error} chatBusy={chatBusy} onFile={handleFile} onSelect={setSelected} onReview={review} onChat={sendEditMessage} onCancel={() => uploadRequest.current?.abort()} onRetry={() => { setError(null); void refreshJobs() }} />
      : <Library jobs={jobs} onOpen={openJob} onFile={handleFile} onDelete={requestDelete} />}
    </main>
    {deleteCandidate && <DeleteDialog job={deleteCandidate} busy={deleteBusy} error={deleteError} onCancel={closeDelete} onConfirm={confirmDelete} />}
  </div>
}
