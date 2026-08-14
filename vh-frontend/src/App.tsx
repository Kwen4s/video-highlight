import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, DragEvent, FormEvent, ReactNode, SVGProps } from 'react'

type View = 'extract' | 'library'
type AssetType = 'original' | 'highlight'
type IconName =
  | 'spark'
  | 'library'
  | 'upload'
  | 'search'
  | 'more'
  | 'play'
  | 'pause'
  | 'send'
  | 'scissors'
  | 'wand'
  | 'check'
  | 'clock'
  | 'film'
  | 'folder'
  | 'grid'
  | 'chevron'
  | 'close'
  | 'minus'
  | 'square'
  | 'plus'

type Message = {
  id: number
  role: 'assistant' | 'user'
  body: string
  meta?: string
}

type Asset = {
  id: number
  title: string
  source: string
  type: AssetType
  duration: string
  resolution: string
  size: string
  createdAt: string
  score?: number
  palette: number
}

const assets: Asset[] = [
  { id: 1, title: '夏日餐桌_品牌广告', source: '项目原片', type: 'original', duration: '04:18', resolution: '4K', size: '1.86 GB', createdAt: '今天 14:32', palette: 1 },
  { id: 2, title: '碰杯之后的笑场', source: '夏日餐桌_品牌广告', type: 'highlight', duration: '00:18', resolution: '1080P', size: '42 MB', createdAt: '今天 14:38', score: 96, palette: 2 },
  { id: 3, title: '日落下的最后一口', source: '夏日餐桌_品牌广告', type: 'highlight', duration: '00:12', resolution: '1080P', size: '31 MB', createdAt: '今天 14:38', score: 93, palette: 3 },
  { id: 4, title: '主厨揭开餐盘', source: '夏日餐桌_品牌广告', type: 'highlight', duration: '00:09', resolution: '1080P', size: '24 MB', createdAt: '今天 14:38', score: 88, palette: 4 },
  { id: 5, title: '城市漫游_VLOG', source: '项目原片', type: 'original', duration: '12:46', resolution: '4K', size: '4.23 GB', createdAt: '昨天 18:06', palette: 5 },
  { id: 6, title: '穿过旧城的蓝色时刻', source: '城市漫游_VLOG', type: 'highlight', duration: '00:24', resolution: '1080P', size: '58 MB', createdAt: '昨天 18:14', score: 91, palette: 6 },
  { id: 7, title: '新品发布会_机位A', source: '项目原片', type: 'original', duration: '38:05', resolution: '1080P', size: '5.14 GB', createdAt: '8月12日', palette: 7 },
  { id: 8, title: '全场掌声与产品亮相', source: '新品发布会_机位A', type: 'highlight', duration: '00:31', resolution: '1080P', size: '72 MB', createdAt: '8月12日', score: 89, palette: 8 },
  { id: 9, title: '创作者访谈_双机剪辑', source: '项目原片', type: 'original', duration: '22:18', resolution: '4K', size: '6.02 GB', createdAt: '8月10日', palette: 1 },
  { id: 10, title: '提问之后的停顿', source: '创作者访谈_双机剪辑', type: 'highlight', duration: '00:16', resolution: '1080P', size: '39 MB', createdAt: '8月10日', score: 92, palette: 2 },
  { id: 11, title: '嘉宾笑着说出答案', source: '创作者访谈_双机剪辑', type: 'highlight', duration: '00:21', resolution: '1080P', size: '48 MB', createdAt: '8月10日', score: 90, palette: 3 },
  { id: 12, title: '散场前的最后一个拥抱', source: '新品发布会_机位A', type: 'highlight', duration: '00:14', resolution: '1080P', size: '34 MB', createdAt: '8月12日', score: 87, palette: 4 },
]

const initialMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    body: '我找到了 6 段情绪与叙事都很完整的镜头，其中 01:42 的碰杯瞬间最适合作为开场。',
    meta: '已分析 04:18 视频',
  },
  {
    id: 2,
    role: 'user',
    body: '成片想更松弛一些，把人物自然互动留长一点。',
  },
  {
    id: 3,
    role: 'assistant',
    body: '可以。我会保留笑场前后的呼吸感，并减少快切。建议把第一段向后延长 2 秒，成片约 52 秒。',
    meta: '等待你的确认',
  },
]

const waveform = [
  9, 15, 21, 13, 28, 34, 18, 42, 31, 17, 12, 26, 46, 38, 22, 15, 29, 51, 39, 24, 18, 44, 57, 33,
  20, 14, 27, 48, 61, 45, 19, 11, 23, 37, 52, 29, 16, 25, 43, 56, 40, 18, 12, 35, 49, 63, 30, 21,
  17, 26, 55, 41, 33, 16, 10, 30, 47, 58, 37, 22, 14, 28, 51, 35, 19, 12, 32, 44, 59, 39, 24, 15,
  25, 46, 53, 31, 18, 11, 29, 41, 62, 44, 27, 17, 23, 49, 56, 36, 20, 13, 31, 45, 52, 34, 18, 10,
]

function Icon({ name, size = 18, ...props }: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  const paths: Record<IconName, ReactNode> = {
    spark: <><path d="m12 3-1.1 4.1a5.2 5.2 0 0 1-3.8 3.8L3 12l4.1 1.1a5.2 5.2 0 0 1 3.8 3.8L12 21l1.1-4.1a5.2 5.2 0 0 1 3.8-3.8L21 12l-4.1-1.1a5.2 5.2 0 0 1-3.8-3.8L12 3Z"/><path d="M5 3v4M3 5h4M19 17v4M17 19h4"/></>,
    library: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M8 4v5"/></>,
    upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M5 20h14"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
    more: <><circle cx="5" cy="12" r="1"/><circle cx="12" cy="12" r="1"/><circle cx="19" cy="12" r="1"/></>,
    play: <path d="m9 7 8 5-8 5V7Z" fill="currentColor" stroke="none"/>,
    pause: <><path d="M9 7v10M15 7v10"/></>,
    send: <><path d="m22 2-7 20-4-9-9-4 20-7Z"/><path d="M22 2 11 13"/></>,
    scissors: <><circle cx="6" cy="7" r="3"/><circle cx="6" cy="17" r="3"/><path d="m8.7 8.4 12.3 6.1M8.7 15.6 21 9.5"/></>,
    wand: <><path d="m15 4 5 5L8 21l-5-5L15 4Z"/><path d="m6 4 .7 2.3L9 7l-2.3.7L6 10l-.7-2.3L3 7l2.3-.7L6 4ZM19 14l.5 1.5L21 16l-1.5.5L19 18l-.5-1.5L17 16l1.5-.5L19 14Z"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    film: <><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M17 9h4M3 15h4M17 15h4"/></>,
    folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5H10l2 2h6.5A2.5 2.5 0 0 1 21 9.5v7A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5v-9Z"/>,
    grid: <><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    close: <path d="m7 7 10 10M17 7 7 17"/>,
    minus: <path d="M5 12h14"/>,
    square: <rect x="7" y="7" width="10" height="10" rx="1"/>,
    plus: <path d="M12 5v14M5 12h14"/>,
  }

  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>
      {paths[name]}
    </svg>
  )
}

function WindowChrome() {
  return (
    <div className="window-chrome">
      <div className="chrome-project"><span className="chrome-dot" />FRAME / 高光工作台</div>
      <div className="window-actions no-drag">
        <button aria-label="最小化" onClick={() => window.desktopWindow?.minimize()}><Icon name="minus" size={15} /></button>
        <button aria-label="最大化" onClick={() => window.desktopWindow?.toggleMaximize()}><Icon name="square" size={14} /></button>
        <button className="window-close" aria-label="关闭" onClick={() => window.desktopWindow?.close()}><Icon name="close" size={15} /></button>
      </div>
    </div>
  )
}

function Sidebar({ activeView, onViewChange }: { activeView: View; onViewChange: (view: View) => void }) {
  return (
    <aside className="sidebar">
      <div className="brand-block">
        <div className="brand-mark"><span /><span /><span /></div>
        <div><strong>FRAME</strong><small>高光工作台</small></div>
      </div>

      <nav className="main-nav" aria-label="主导航">
        <span className="nav-label">工作区</span>
        <button className={activeView === 'extract' ? 'active' : ''} onClick={() => onViewChange('extract')}>
          <Icon name="spark" /><span>高光提取</span><i>01</i>
        </button>
        <button className={activeView === 'library' ? 'active' : ''} onClick={() => onViewChange('library')}>
          <Icon name="library" /><span>资源库</span><i>02</i>
        </button>
      </nav>

      <div className="storage-card">
        <div className="storage-title"><span>本地存储</span><b>68%</b></div>
        <div className="storage-track"><span /></div>
        <p>已使用 136 GB / 200 GB</p>
      </div>

      <div className="profile-row">
        <div className="avatar">T</div>
        <div><b>Tamako</b><small>创作者空间</small></div>
        <button aria-label="更多"><Icon name="more" /></button>
      </div>
    </aside>
  )
}

function StatusPill({ status, progress }: { status: 'processing' | 'done'; progress: number }) {
  return status === 'processing' ? (
    <div className="status-pill processing"><span className="spinner" />正在分析 {progress}%</div>
  ) : (
    <div className="status-pill complete"><Icon name="check" size={14} />分析完成</div>
  )
}

function UploadButton({ onChange, compact = false }: { onChange: (event: ChangeEvent<HTMLInputElement>) => void; compact?: boolean }) {
  return (
    <label className={compact ? 'button secondary compact' : 'button primary'}>
      <Icon name="upload" size={17} />{compact ? '替换视频' : '导入视频'}
      <input type="file" accept="video/*" onChange={onChange} hidden />
    </label>
  )
}

function Timeline({ processing }: { processing: boolean }) {
  return (
    <div className="timeline-wrap">
      <div className="timeline-head"><span>00:00</span><span>01:00</span><span>02:00</span><span>03:00</span><span>04:18</span></div>
      <div className="timeline">
        <div className="waveform">
          {waveform.map((height, index) => <i key={index} style={{ height: `${height}%` }} />)}
        </div>
        <span className="highlight-range range-a" /><span className="highlight-range range-b" /><span className="highlight-range range-c" />
        <span className="playhead"><b /></span>
        {processing && <span className="scanline" />}
      </div>
      <div className="timeline-legend"><span><i className="legend-highlight" />模型推荐</span><span><i className="legend-playhead" />当前画面 01:46</span></div>
    </div>
  )
}

function ChatPanel({ onRegenerate }: { onRegenerate: () => void }) {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [draft, setDraft] = useState('')
  const [thinking, setThinking] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = scrollRef.current
    if (container) container.scrollTo({ top: container.scrollHeight, behavior: 'smooth' })
  }, [messages, thinking])

  const submit = (event: FormEvent) => {
    event.preventDefault()
    const value = draft.trim()
    if (!value || thinking) return
    setMessages((current) => [...current, { id: Date.now(), role: 'user', body: value }])
    setDraft('')
    setThinking(true)
    window.setTimeout(() => {
      const wantsRegenerate = /重做|重新|生成|节奏|松弛/.test(value)
      setMessages((current) => [...current, {
        id: Date.now() + 1,
        role: 'assistant',
        body: wantsRegenerate
          ? '收到。我会降低切换频率，优先保留人物表情和动作落点。已按这个方向生成一个新版本，你可以在左侧时间轴预览。'
          : '已记下这个修改。我会只调整当前选中的高光段，不影响其他片段。',
        meta: wantsRegenerate ? '新版本预计 52 秒' : '修改已应用',
      }])
      if (wantsRegenerate) onRegenerate()
      setThinking(false)
    }, 750)
  }

  const runSuggestion = (text: string) => {
    setDraft(text)
  }

  return (
    <aside className="chat-panel">
      <div className="panel-heading">
        <div><span className="ai-orb"><Icon name="spark" size={15} /></span><div><b>AI 剪辑搭档</b><small><i />随时可以调整</small></div></div>
        <button aria-label="更多"><Icon name="more" /></button>
      </div>

      <div className="chat-scroll" ref={scrollRef}>
        {messages.map((message) => (
          <div key={message.id} className={`message ${message.role}`}>
            {message.role === 'assistant' && <div className="message-avatar"><Icon name="spark" size={13} /></div>}
            <div className="message-content"><p>{message.body}</p>{message.meta && <small>{message.meta}</small>}</div>
          </div>
        ))}
        {thinking && <div className="message assistant"><div className="message-avatar"><Icon name="spark" size={13} /></div><div className="typing"><i /><i /><i /></div></div>}
      </div>

      <div className="suggestions">
        <button onClick={() => runSuggestion('重新生成一个更松弛的版本')}><Icon name="wand" size={14} />重做得更松弛</button>
        <button onClick={() => runSuggestion('把第一段结尾延长 2 秒')}><Icon name="scissors" size={14} />结尾延长 2 秒</button>
      </div>
      <form className="chat-input" onSubmit={submit}>
        <textarea value={draft} onChange={(event) => setDraft(event.target.value)} onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault()
            event.currentTarget.form?.requestSubmit()
          }
        }} placeholder="描述你想要的节奏、情绪或修改…" rows={2} />
        <div><span>Enter 发送 · Shift + Enter 换行</span><button type="submit" aria-label="发送" disabled={!draft.trim() || thinking}><Icon name="send" size={16} /></button></div>
      </form>
    </aside>
  )
}

function ExtractView({ onUpload, fileName, status, progress, onRegenerate }: {
  onUpload: (event: ChangeEvent<HTMLInputElement>) => void
  fileName: string
  status: 'processing' | 'done'
  progress: number
  onRegenerate: () => void
}) {
  const [playing, setPlaying] = useState(false)
  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) {
      const transfer = new DataTransfer()
      transfer.items.add(file)
      onUpload({ target: { files: transfer.files } } as ChangeEvent<HTMLInputElement>)
    }
  }

  return (
    <div className="view extract-view">
      <header className="view-header extract-header">
        <div><p className="eyebrow">HIGHLIGHT EXTRACTION</p><h1>上传原片，提取高光</h1></div>
        <div className="header-actions"><span className="autosave"><i />已自动保存</span><UploadButton onChange={onUpload} /></div>
      </header>

      <section className="workspace-grid">
        <div className="review-workspace" onDragOver={(event) => event.preventDefault()} onDrop={handleDrop}>
          <div className="source-bar">
            <div className="source-icon"><Icon name="film" /></div>
            <div className="source-copy"><span>{fileName}</span><small>MP4 · 4K · 04:18 · 1.86 GB</small></div>
            <StatusPill status={status} progress={progress} />
            <UploadButton onChange={onUpload} compact />
            <button className="icon-button" aria-label="更多"><Icon name="more" /></button>
          </div>

          <div className={`video-stage ${status === 'processing' ? 'is-processing' : ''}`}>
            <div className="stage-grain" />
            <div className="frame-caption"><span>FRAME 0142</span><b>SUMMER / TABLE</b></div>
            <button className="play-button" onClick={() => setPlaying((value) => !value)} aria-label={playing ? '暂停' : '播放'}><Icon name={playing ? 'pause' : 'play'} size={24} /></button>
            <div className="stage-bottom"><span>01:46</span><div className="stage-progress"><i /></div><span>04:18</span><button>1×</button></div>
            {status === 'processing' && <div className="analysis-overlay"><span className="analysis-ring"><b>{progress}</b>%</span><p>模型正在理解人物、对白与情绪起伏</p></div>}
          </div>
          <Timeline processing={status === 'processing'} />
        </div>
        <ChatPanel onRegenerate={onRegenerate} />
      </section>

    </div>
  )
}

function AssetCard({ asset }: { asset: Asset }) {
  const [playing, setPlaying] = useState(false)
  return (
    <article className="asset-card">
      <div className={`asset-poster thumb-${asset.palette}`}>
        <div className="poster-index">F/{String(asset.id).padStart(3, '0')}</div>
        <span className={`asset-badge ${asset.type}`}>{asset.type === 'original' ? '原片' : '高光'}</span>
        <button className="asset-play" onClick={() => setPlaying((value) => !value)} aria-label={playing ? '暂停' : '播放'}><Icon name={playing ? 'pause' : 'play'} size={20} /></button>
        <span className="asset-duration">{asset.duration}</span>
        {asset.score && <span className="asset-score"><Icon name="spark" size={12} />{asset.score}</span>}
      </div>
      <div className="asset-info">
        <div className="asset-title"><div><h3>{asset.title}</h3><p>{asset.type === 'original' ? '独立原始素材' : `来源 · ${asset.source}`}</p></div><button aria-label="更多"><Icon name="more" /></button></div>
        <div className="asset-meta"><span>{asset.resolution}</span><span>{asset.size}</span><span>{asset.createdAt}</span></div>
      </div>
    </article>
  )
}

function LibraryView({ onUpload }: { onUpload: (event: ChangeEvent<HTMLInputElement>) => void }) {
  const [filter, setFilter] = useState<'all' | AssetType>('all')
  const [query, setQuery] = useState('')
  const visibleAssets = useMemo(() => assets.filter((asset) => {
    const matchesFilter = filter === 'all' || asset.type === filter
    const matchesQuery = `${asset.title}${asset.source}`.toLowerCase().includes(query.toLowerCase())
    return matchesFilter && matchesQuery
  }), [filter, query])

  return (
    <div className="view library-view">
      <header className="view-header library-header">
        <div><p className="eyebrow">ASSET LIBRARY</p><h1>资源库</h1><p>原片与模型提取的高光统一归档，随时预览、复核和继续编辑。</p></div>
        <UploadButton onChange={onUpload} />
      </header>

      <section className="library-summary">
        <div className="summary-lead"><span className="summary-icon"><Icon name="folder" /></span><div><small>全部资产</small><b>12</b></div></div>
        <div><small>原片</small><b>4</b><span>+1 本周</span></div>
        <div><small>高光片段</small><b>8</b><span>平均评分 91</span></div>
        <div><small>总时长</small><b>1h 12m</b><span>已节省约 46 分钟</span></div>
      </section>

      <section className="library-content">
        <div className="library-toolbar">
          <div className="filter-tabs" role="tablist">
            <button className={filter === 'all' ? 'active' : ''} onClick={() => setFilter('all')}>全部 <span>12</span></button>
            <button className={filter === 'original' ? 'active' : ''} onClick={() => setFilter('original')}>原片 <span>4</span></button>
            <button className={filter === 'highlight' ? 'active' : ''} onClick={() => setFilter('highlight')}>高光片段 <span>8</span></button>
          </div>
          <div className="toolbar-actions">
            <label className="search-box"><Icon name="search" size={16} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索片名或来源" /></label>
            <button className="sort-button">最近创建 <Icon name="chevron" size={14} /></button>
            <button className="grid-button" aria-label="网格视图"><Icon name="grid" size={16} /></button>
          </div>
        </div>

        {visibleAssets.length ? (
          <div className="asset-grid">{visibleAssets.map((asset) => <AssetCard key={asset.id} asset={asset} />)}</div>
        ) : (
          <div className="empty-library"><Icon name="search" size={28} /><h3>没有找到相关资产</h3><p>换一个关键词，或查看全部内容。</p></div>
        )}
      </section>
    </div>
  )
}

export default function App() {
  const [activeView, setActiveView] = useState<View>('extract')
  const [fileName, setFileName] = useState('夏日餐桌_品牌广告_v08.mp4')
  const [status, setStatus] = useState<'processing' | 'done'>('done')
  const [progress, setProgress] = useState(100)

  useEffect(() => {
    if (status !== 'processing') return
    const timer = window.setInterval(() => {
      setProgress((value) => {
        if (value >= 100) {
          window.clearInterval(timer)
          setStatus('done')
          return 100
        }
        return Math.min(100, value + 4)
      })
    }, 120)
    return () => window.clearInterval(timer)
  }, [status])

  const startProcessing = () => {
    setProgress(8)
    setStatus('processing')
  }

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setActiveView('extract')
    startProcessing()
  }

  return (
    <div className="app-shell">
      <WindowChrome />
      <Sidebar activeView={activeView} onViewChange={setActiveView} />
      <main className="main-surface">
        {activeView === 'extract' ? (
          <ExtractView onUpload={handleUpload} fileName={fileName} status={status} progress={progress} onRegenerate={startProcessing} />
        ) : (
          <LibraryView onUpload={handleUpload} />
        )}
      </main>
    </div>
  )
}
