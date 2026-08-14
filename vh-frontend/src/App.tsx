import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent, ReactNode, SVGProps } from 'react'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '')
const VIDEO_EXTENSIONS = ['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v']

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
  clip_url: string
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
  source_url: string
  error_message: string | null
  result: DetectionResult | null
}

type UploadState = {
  phase: 'idle' | 'preparing' | 'uploading'
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
  const response = await fetch(`${API_BASE}${path}`, init)
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
    throw new Error('无法连接本地后端；如果程序刚刚更新，请重启 npm run dev 后再试')
  }
  if (!response.ok) {
    const payload = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(payload?.detail || `请求失败（${response.status}）`)
  }
}

function uploadVideo(
  jobId: string,
  file: File,
  onProgress: (value: number) => void,
  register: (request: XMLHttpRequest | null) => void,
): Promise<Job> {
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
      if (request.status >= 200 && request.status < 300) resolve(request.response as Job)
      else reject(new Error(request.response?.detail || `上传失败（${request.status}）`))
    }
    request.onerror = () => { register(null); reject(new Error('无法连接本地后端服务')) }
    request.onabort = () => { register(null); reject(new Error('上传已取消')) }
    request.send(form)
  })
}

function apiMediaUrl(path: string) {
  return path.startsWith('http') ? path : `${API_BASE}${path}`
}

function createJobId() {
  return `job_${crypto.randomUUID().replace(/-/g, '').slice(0, 16)}`
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
    <div className="sidebar-metric"><span>本机任务</span><strong>{String(jobs.length).padStart(2, '0')}</strong><small>{completed} 个已完成</small></div>
    <div className={`service-state ${online ? 'online' : ''}`}><i /><div><b>{online ? '本地服务在线' : '本地服务离线'}</b><span>{online ? 'API · 127.0.0.1:8000' : '请启动 FastAPI 后端'}</span></div></div>
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
    <p>视频会以流式方式传给本地后端；任务目录、原片和高光片段都留在这台电脑。</p>
    <button className="button primary" onClick={() => inputRef.current?.click()}><Icon name="film" size={17} />选择视频文件</button>
    <span>MP4 · MOV · MKV · WEBM · AVI · 最大 20 GB</span>
  </div>
}

function ProgressPanel({ upload, job, onCancel }: { upload: UploadState; job: Job | null; onCancel: () => void }) {
  const uploading = upload.phase !== 'idle'
  const label = upload.phase === 'preparing' ? '正在创建本机任务目录' : upload.phase === 'uploading' ? `正在上传 ${upload.progress}%` : job ? statusCopy[job.status] : ''
  const number = upload.phase === 'uploading' ? upload.progress : job?.status === 'completed' ? 100 : null
  return <section className={`progress-panel ${job?.status || upload.phase}`}>
    <div className="progress-heading"><div><span className="pulse" /><div><b>{label}</b><small>{uploading ? upload.fileName : job?.original_name}</small></div></div>{upload.phase === 'uploading' && <button onClick={onCancel}><Icon name="x" size={15} />取消上传</button>}</div>
    <div className={`progress-track ${job?.status === 'processing' || job?.status === 'queued' || upload.phase === 'preparing' ? 'indeterminate' : ''}`}><i style={{ width: `${number ?? 38}%` }} /></div>
    <div className="phase-rail">
      <span className="done"><i>1</i>创建目录</span>
      <span className={upload.phase === 'uploading' ? 'current' : job ? 'done' : ''}><i>2</i>保存原片</span>
      <span className={job?.status === 'queued' || job?.status === 'processing' ? 'current' : job?.status === 'completed' ? 'done' : ''}><i>3</i>Agent 分析</span>
      <span className={job?.status === 'completed' ? 'done' : ''}><i>4</i>等待复核</span>
    </div>
  </section>
}

function Player({ job, selected }: { job: Job; selected: Highlight | null }) {
  const source = selected ? selected.clip_url : job.source_url
  const highlights = job.result?.highlights || []
  const sourceDuration = job.result?.video.duration_sec || 1
  const videoRef = useRef<HTMLVideoElement>(null)
  useEffect(() => {
    const player = videoRef.current
    if (!player) return
    player.src = apiMediaUrl(source)
    player.load()
    return () => {
      player.pause()
      player.removeAttribute('src')
      player.load()
    }
  }, [source])
  return <section className="player-card">
    <div className="player-top"><div><span>VIDEO MONITOR · {selected ? 'HIGHLIGHT' : 'SOURCE'}</span><b>{selected?.description || job.original_name}</b></div><span className="media-chip">{selected ? `${formatTime(selected.start_sec)} — ${formatTime(selected.end_sec)}` : formatBytes(job.size_bytes)}</span></div>
    <div className="video-frame">
      <div className="video-stage"><video ref={videoRef} key={source} controls preload="metadata" /></div>
      <div className="monitor-timeline" aria-label="完整原片高光时间轴">
        <div className="timeline-caption"><span>完整原片 · 高光轨道</span><b>{formatTime(sourceDuration)}</b></div>
        <div className="timeline-track">
          {highlights.map((item) => {
            const start = Math.max(0, Math.min(100, (item.start_sec / sourceDuration) * 100))
            const end = Math.max(start, Math.min(100, (item.end_sec / sourceDuration) * 100))
            return <span
              key={item.highlight_id}
              className={selected?.highlight_id === item.highlight_id ? 'timeline-highlight active' : 'timeline-highlight'}
              style={{ left: `${start}%`, width: `${end - start}%` }}
              title={`${item.description} · ${formatTime(item.start_sec)} — ${formatTime(item.end_sec)}`}
            />
          })}
        </div>
        <div className="timeline-scale"><span>00:00</span><span>荧光绿区域为高光片段</span><span>{formatTime(sourceDuration)}</span></div>
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

function HighlightAssistant({ job, selected }: { job: Job; selected: Highlight | null }) {
  const focus = selected ? `「${selected.description}」` : '完整原片'
  return <aside className="assistant-panel">
    <div className="assistant-head">
      <div><p className="overline">AI EDIT ASSISTANT</p><h2>高光编辑助手</h2></div>
      <span className="assistant-state"><i />就绪</span>
    </div>
    <div className="assistant-thread" role="log" aria-label="高光编辑助手对话示例">
      <div className="assistant-context"><span>当前上下文</span><b>{focus}</b><small>{job.result?.highlights.length || 0} 个高光候选已载入</small></div>
      <div className="message assistant-message">
        <span className="message-avatar"><Icon name="spark" size={15} /></span>
        <div><small>EDIT AI</small><p>我正在查看{focus}。你可以让我调整入点、缩短时长，或者重写标题和说明。</p></div>
      </div>
      <div className="message user-message">
        <div><small>YOU</small><p>把开头再收紧一点，保留最后的定格。</p></div>
      </div>
      <div className="message assistant-message">
        <span className="message-avatar"><Icon name="spark" size={15} /></span>
        <div><small>EDIT AI</small><p>可以。我会把入点后移约 0.8 秒，并保持出点不变。应用前会先生成预览。</p></div>
      </div>
      <div className="assistant-suggestions"><span>快捷指令</span><div><button type="button">缩短片段</button><button type="button">保留高潮</button><button type="button">改写标题</button></div></div>
    </div>
    <div className="assistant-composer" aria-label="AI 编辑指令输入示意">
      <span>描述你想怎样调整这个高光…</span>
      <button type="button" aria-label="发送编辑指令" disabled>→</button>
    </div>
  </aside>
}

function Workspace({ upload, job, selected, error, onFile, onSelect, onReview, onCancel, onRetry }: { upload: UploadState; job: Job | null; selected: Highlight | null; error: string | null; onFile: (file: File) => void; onSelect: (item: Highlight | null) => void; onReview: (item: Highlight, status: ReviewStatus) => void; onCancel: () => void; onRetry: () => void }) {
  const busy = upload.phase !== 'idle' || job?.status === 'queued' || job?.status === 'processing'
  return <div className="view workspace-view">
    <header className="view-header"><div><p className="overline">LOCAL VIDEO INTELLIGENCE</p><h1>从原片到值得留下的一刻。</h1><p>导入视频，后台完成保存与高光提取，再由你做最后判断。</p></div><UploadDropzone compact onFile={onFile} /></header>
    {error && <div className="error-banner"><span><Icon name="close" size={16} /></span><div><b>流程没有完成</b><p>{error}</p></div><button onClick={onRetry}><Icon name="retry" size={15} />重新载入</button></div>}
    {!job && upload.phase === 'idle' && <UploadDropzone onFile={onFile} />}
    {busy && <ProgressPanel upload={upload} job={job} onCancel={onCancel} />}
    {job?.status === 'failed' && <div className="failed-state"><Icon name="close" size={22} /><div><h2>Agent 未能完成分析</h2><p>{job.error_message || '请查看任务日志后重试。原视频已经安全保存。'}</p></div></div>}
    {job?.status === 'completed' && <div className="review-grid"><HighlightList job={job} selectedId={selected?.highlight_id || null} onSelect={onSelect} onReview={onReview} /><Player job={job} selected={selected} /><HighlightAssistant job={job} selected={selected} /></div>}
  </div>
}

function Library({ jobs, onOpen, onFile, onDelete }: { jobs: Job[]; onOpen: (job: Job) => void; onFile: (file: File) => void; onDelete: (job: Job) => void }) {
  return <div className="view library-view">
    <header className="view-header"><div><p className="overline">LOCAL ARCHIVE</p><h1>任务归档</h1><p>每次导入都对应一个独立目录，原片、结果与复核状态一起保存。</p></div><UploadDropzone compact onFile={onFile} /></header>
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
      {!jobs.length && <div className="empty-archive"><Icon name="folder" size={25} /><h2>还没有导入记录</h2><p>从第一条视频开始建立本地高光库。</p></div>}
    </section>
  </div>
}

function DeleteDialog({ job, busy, error, onCancel, onConfirm }: { job: Job; busy: boolean; error: string | null; onCancel: () => void; onConfirm: () => void }) {
  return <div className="dialog-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget && !busy) onCancel() }}>
    <section className="delete-dialog" role="dialog" aria-modal="true" aria-labelledby="delete-title">
      <div className="dialog-mark"><Icon name="trash" size={21} /></div>
      <p className="overline">DELETE LOCAL TASK</p>
      <h2 id="delete-title">删除这个任务？</h2>
      <p>任务记录、原视频、高光片段和日志都会从本机永久删除，此操作无法撤销。</p>
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
  const uploadRequest = useRef<XMLHttpRequest | null>(null)

  const refreshJobs = useCallback(async () => {
    try {
      const records = await requestJson<Job[]>('/api/jobs')
      setJobs(records)
      setOnline(true)
      return records
    } catch {
      setOnline(false)
      return []
    }
  }, [])

  useEffect(() => { void refreshJobs() }, [refreshJobs])

  useEffect(() => {
    if (!job || !['queued', 'processing'].includes(job.status)) return
    let cancelled = false
    const timer = window.setInterval(async () => {
      try {
        const latest = await requestJson<Job>(`/api/jobs/${job.job_id}`)
        if (cancelled) return
        setJob(latest)
        setOnline(true)
        if (!['queued', 'processing'].includes(latest.status)) {
          window.clearInterval(timer)
          void refreshJobs()
        }
      } catch (reason) {
        if (!cancelled) setError(reason instanceof Error ? reason.message : '任务状态查询失败')
      }
    }, 1800)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [job?.job_id, job?.status, refreshJobs])

  useEffect(() => {
    if (!job?.result?.highlights.length) setSelected(null)
    else if (selected) setSelected(job.result.highlights.find((item) => item.highlight_id === selected.highlight_id) || null)
  }, [job?.updated_at])

  const handleFile = async (file: File) => {
    const extension = file.name.includes('.') ? file.name.slice(file.name.lastIndexOf('.')).toLowerCase() : ''
    if (!VIDEO_EXTENSIONS.includes(extension)) { setError('请选择 MP4、MOV、MKV、WEBM、AVI 或 M4V 视频。'); return }
    if (file.size > 20 * 1024 ** 3) { setError('视频不能超过 20 GB。'); return }
    if (!window.videoImports) { setError('任务目录只能由 Electron 桌面端创建，请在桌面应用中导入。'); return }

    const jobId = createJobId()
    setView('workspace')
    setJob(null)
    setSelected(null)
    setError(null)
    setUpload({ phase: 'preparing', progress: 0, fileName: file.name })
    try {
      await window.videoImports.prepare({ jobId, fileName: file.name })
      setUpload({ phase: 'uploading', progress: 0, fileName: file.name })
      const created = await uploadVideo(jobId, file, (progress) => setUpload({ phase: 'uploading', progress, fileName: file.name }), (request) => { uploadRequest.current = request })
      setJob(created)
      setUpload({ phase: 'idle', progress: 0 })
      setOnline(true)
      await refreshJobs()
    } catch (reason) {
      setUpload({ phase: 'idle', progress: 0 })
      setError(reason instanceof Error ? reason.message : '视频导入失败')
    }
  }

  const review = async (item: Highlight, reviewStatus: ReviewStatus) => {
    if (!job) return
    try {
      const updated = await requestJson<Job>(`/api/jobs/${job.job_id}/highlights/${item.highlight_id}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status: reviewStatus }),
      })
      setJob(updated)
      setJobs((current) => current.map((record) => record.job_id === updated.job_id ? updated : record))
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '复核状态保存失败')
    }
  }

  const openJob = (record: Job) => { setJob(record); setSelected(null); setError(null); setView('workspace') }
  const requestDelete = (record: Job) => { setDeleteCandidate(record); setDeleteError(null) }
  const closeDelete = () => { if (!deleteBusy) { setDeleteCandidate(null); setDeleteError(null) } }
  const confirmDelete = async () => {
    if (!deleteCandidate) return
    setDeleteBusy(true)
    setDeleteError(null)
    try {
      await requestEmpty(`/api/jobs/${deleteCandidate.job_id}`, { method: 'DELETE' })
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
      ? <Workspace upload={upload} job={job} selected={activeHighlight} error={error} onFile={handleFile} onSelect={setSelected} onReview={review} onCancel={() => uploadRequest.current?.abort()} onRetry={() => { setError(null); void refreshJobs() }} />
      : <Library jobs={jobs} onOpen={openJob} onFile={handleFile} onDelete={requestDelete} />}
    </main>
    {deleteCandidate && <DeleteDialog job={deleteCandidate} busy={deleteBusy} error={deleteError} onCancel={closeDelete} onConfirm={confirmDelete} />}
  </div>
}
