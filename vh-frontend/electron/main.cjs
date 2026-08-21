const { app, BrowserWindow, ipcMain, net, protocol } = require('electron')
const { copyFile, cp, mkdir, readFile, readdir, rename, rm, stat, writeFile } = require('node:fs/promises')
const { randomUUID } = require('node:crypto')
const path = require('node:path')
const { pathToFileURL } = require('node:url')

const isDevelopment = process.argv.includes('--dev')
const JOB_ID_PATTERN = /^job_[A-Za-z0-9_-]{8,48}$/
const VIDEO_EXTENSIONS = new Set(['.mp4', '.mov', '.mkv', '.webm', '.avi', '.m4v'])
const MEDIA_SCHEME = 'vh-media'
const DEFAULT_DEMO_VERSION = 1
const DEFAULT_DEMO_JOB_IDS = ['job_demo_citypulse', 'job_demo_launchfilm']

protocol.registerSchemesAsPrivileged([{
  scheme: MEDIA_SCHEME,
  privileges: { standard: true, secure: true, supportFetchAPI: true, stream: true },
}])

function requireJobId(value) {
  if (typeof value !== 'string' || !JOB_ID_PATTERN.test(value)) throw new Error('无效的任务编号')
  return value
}

function getJobsRoot() {
  return path.join(app.getPath('userData'), 'video-library', 'jobs')
}

function getLibraryRoot() {
  return path.dirname(getJobsRoot())
}

function getDemoMarkerPath() {
  return path.join(getLibraryRoot(), 'default-demos.json')
}

function getBundledDemoJobsRoot() {
  return path.resolve(__dirname, '..', 'video-data', 'jobs')
}

function getJobDirectory(jobId) {
  return path.join(getJobsRoot(), requireJobId(jobId))
}

function getTaskPath(jobId) {
  return path.join(getJobDirectory(jobId), 'task.json')
}

function localSourceUrl(jobId) {
  return `${MEDIA_SCHEME}://jobs/${requireJobId(jobId)}/source`
}

async function readStoredJob(jobId) {
  const payload = await readFile(getTaskPath(jobId), 'utf8')
  const parsed = JSON.parse(payload)
  if (parsed.job_id !== jobId || typeof parsed.local_source_name !== 'string') {
    throw new Error('本地任务数据无效')
  }
  return parsed
}

function publicJob(stored) {
  const { local_source_name: _localSourceName, ...task } = stored
  return { ...task, source_url: localSourceUrl(task.job_id) }
}

async function writeStoredJob(jobId, task) {
  const taskPath = getTaskPath(jobId)
  const temporaryPath = `${taskPath}.${randomUUID()}.tmp`
  await writeFile(temporaryPath, JSON.stringify(task, null, 2), 'utf8')
  await rename(temporaryPath, taskPath)
}

async function hasInitializedDefaultDemos() {
  try {
    const marker = JSON.parse(await readFile(getDemoMarkerPath(), 'utf8'))
    return marker.version === DEFAULT_DEMO_VERSION
  } catch {
    return false
  }
}

async function buildDemoTask(templateDirectory, jobId, createdAt) {
  const result = JSON.parse(await readFile(path.join(templateDirectory, 'result.json'), 'utf8'))
  if (result?.job_id !== jobId || !Array.isArray(result?.highlights)) {
    throw new Error(`Demo 任务数据无效：${jobId}`)
  }

  const sourceDirectory = path.join(templateDirectory, 'source')
  const sourceEntries = await readdir(sourceDirectory, { withFileTypes: true })
  const sourceEntry = sourceEntries.find((entry) => (
    entry.isFile() && VIDEO_EXTENSIONS.has(path.extname(entry.name).toLowerCase())
  ))
  if (!sourceEntry) throw new Error(`Demo 原片不存在：${jobId}`)

  const sourceStats = await stat(path.join(sourceDirectory, sourceEntry.name))
  const originalName = typeof result?.video?.title === 'string'
    ? path.basename(result.video.title)
    : sourceEntry.name
  return {
    job_id: jobId,
    status: 'completed',
    original_name: originalName,
    content_type: 'video/mp4',
    size_bytes: sourceStats.size,
    language: 'zh',
    created_at: createdAt,
    updated_at: createdAt,
    session_expires_at: null,
    revision: 0,
    error_message: null,
    result: {
      ...result,
      highlights: result.highlights.map((highlight) => ({
        ...highlight,
        review_status: highlight.review_status || 'pending',
      })),
    },
    messages: [],
    local_source_name: path.join('source', sourceEntry.name),
  }
}

async function installDefaultDemo(jobId, index) {
  const targetDirectory = getJobDirectory(jobId)
  try {
    await readStoredJob(jobId)
    return
  } catch (error) {
    try {
      const targetStats = await stat(targetDirectory)
      if (targetStats.isDirectory()) {
        throw new Error(`Demo 任务目录已存在但数据无效：${jobId}`, { cause: error })
      }
    } catch (targetError) {
      if (targetError?.code !== 'ENOENT') throw targetError
    }
  }

  const templateDirectory = path.join(getBundledDemoJobsRoot(), jobId)
  const createdAt = new Date(Date.now() - index * 1000).toISOString()
  const task = await buildDemoTask(templateDirectory, jobId, createdAt)
  const stagingDirectory = path.join(getJobsRoot(), `.demo-${jobId}-${randomUUID()}`)
  try {
    await cp(templateDirectory, stagingDirectory, { recursive: true, errorOnExist: true })
    await writeFile(path.join(stagingDirectory, 'task.json'), JSON.stringify(task, null, 2), 'utf8')
    await rename(stagingDirectory, targetDirectory)
  } catch (error) {
    await rm(stagingDirectory, { recursive: true, force: true })
    throw error
  }
}

async function ensureDefaultDemos() {
  await mkdir(getJobsRoot(), { recursive: true })
  if (await hasInitializedDefaultDemos()) return

  for (const [index, jobId] of DEFAULT_DEMO_JOB_IDS.entries()) {
    await installDefaultDemo(jobId, index)
  }

  const markerPath = getDemoMarkerPath()
  const temporaryPath = `${markerPath}.${randomUUID()}.tmp`
  await writeFile(temporaryPath, JSON.stringify({
    version: DEFAULT_DEMO_VERSION,
    job_ids: DEFAULT_DEMO_JOB_IDS,
    initialized_at: new Date().toISOString(),
  }, null, 2), 'utf8')
  await rename(temporaryPath, markerPath)
}

async function importSource(input) {
  const jobId = requireJobId(input?.jobId)
  const sourcePath = typeof input?.sourcePath === 'string' ? path.resolve(input.sourcePath) : ''
  const originalName = path.basename(typeof input?.originalName === 'string' ? input.originalName : '')
  const extension = path.extname(originalName).toLowerCase()
  if (!path.isAbsolute(sourcePath) || !VIDEO_EXTENSIONS.has(extension)) {
    throw new Error('请选择受支持的本地视频文件')
  }
  const sourceStats = await stat(sourcePath)
  if (!sourceStats.isFile()) throw new Error('选择的视频文件不存在')

  const jobDirectory = getJobDirectory(jobId)
  await mkdir(getJobsRoot(), { recursive: true })
  await mkdir(jobDirectory, { recursive: false })
  const storedName = `original${extension}`
  const targetPath = path.join(jobDirectory, storedName)
  try {
    await copyFile(sourcePath, targetPath)
    const createdAt = new Date().toISOString()
    const stored = {
      job_id: jobId,
      status: 'queued',
      original_name: originalName,
      content_type: typeof input?.contentType === 'string' ? input.contentType : 'application/octet-stream',
      size_bytes: sourceStats.size,
      language: input?.language === 'en' ? 'en' : 'zh',
      created_at: createdAt,
      updated_at: createdAt,
      session_expires_at: null,
      revision: 0,
      error_message: null,
      result: null,
      messages: [],
      local_source_name: storedName,
    }
    await writeStoredJob(jobId, stored)
    return publicJob(stored)
  } catch (error) {
    const resolvedJobDirectory = path.resolve(jobDirectory)
    if (path.dirname(resolvedJobDirectory) === path.resolve(getJobsRoot())) {
      await rm(resolvedJobDirectory, { recursive: true, force: true })
    }
    throw error
  }
}

async function saveJob(input) {
  const jobId = requireJobId(input?.job_id)
  const current = await readStoredJob(jobId)
  const messages = Array.isArray(input?.messages) ? input.messages.slice(-200) : current.messages
  const stored = {
    ...current,
    status: input.status,
    updated_at: input.updated_at,
    session_expires_at: input.session_expires_at ?? null,
    revision: Number.isInteger(input.revision) ? input.revision : current.revision,
    error_message: input.error_message ?? null,
    result: input.result ?? null,
    messages,
    job_id: current.job_id,
    original_name: current.original_name,
    content_type: current.content_type,
    size_bytes: current.size_bytes,
    language: current.language,
    created_at: current.created_at,
    local_source_name: current.local_source_name,
  }
  await writeStoredJob(jobId, stored)
  return publicJob(stored)
}

async function listJobs() {
  await ensureDefaultDemos()
  const entries = await readdir(getJobsRoot(), { withFileTypes: true })
  const tasks = await Promise.all(entries.filter((entry) => entry.isDirectory() && JOB_ID_PATTERN.test(entry.name)).map(async (entry) => {
    try {
      return publicJob(await readStoredJob(entry.name))
    } catch {
      return null
    }
  }))
  return tasks.filter(Boolean).sort((left, right) => right.created_at.localeCompare(left.created_at))
}

async function deleteLocalJob(jobIdValue) {
  const jobDirectory = path.resolve(getJobDirectory(jobIdValue))
  if (path.dirname(jobDirectory) !== path.resolve(getJobsRoot())) throw new Error('本地任务目录无效')
  await rm(jobDirectory, { recursive: true, force: true })
}

function registerIpcHandlers() {
  ipcMain.on('window:minimize', (event) => BrowserWindow.fromWebContents(event.sender)?.minimize())
  ipcMain.on('window:toggle-maximize', (event) => {
    const window = BrowserWindow.fromWebContents(event.sender)
    if (!window) return
    if (window.isMaximized()) window.unmaximize()
    else window.maximize()
  })
  ipcMain.on('window:close', (event) => BrowserWindow.fromWebContents(event.sender)?.close())
  ipcMain.handle('library:import-source', (_event, input) => importSource(input))
  ipcMain.handle('library:save-job', (_event, input) => saveJob(input))
  ipcMain.handle('library:list-jobs', () => listJobs())
  ipcMain.handle('library:delete-job', (_event, jobId) => deleteLocalJob(jobId))
}

function registerMediaProtocol() {
  protocol.handle(MEDIA_SCHEME, async (request) => {
    try {
      const url = new URL(request.url)
      const parts = url.pathname.split('/').filter(Boolean)
      if (url.host !== 'jobs' || parts.length !== 2 || parts[1] !== 'source') {
        return new Response('Not found', { status: 404 })
      }
      const stored = await readStoredJob(requireJobId(parts[0]))
      const jobDirectory = path.resolve(getJobDirectory(stored.job_id))
      const sourcePath = path.resolve(jobDirectory, stored.local_source_name)
      const relativeSourcePath = path.relative(jobDirectory, sourcePath)
      if (
        !relativeSourcePath
        || relativeSourcePath === '..'
        || relativeSourcePath.startsWith(`..${path.sep}`)
        || path.isAbsolute(relativeSourcePath)
      ) return new Response('Forbidden', { status: 403 })
      return net.fetch(pathToFileURL(sourcePath).toString(), { headers: request.headers })
    } catch {
      return new Response('Not found', { status: 404 })
    }
  })
}

function createWindow() {
  const window = new BrowserWindow({
    width: 1480,
    height: 940,
    minWidth: 1080,
    minHeight: 700,
    backgroundColor: '#0b0d0d',
    titleBarStyle: 'hidden',
    titleBarOverlay: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  })

  if (isDevelopment) window.loadURL('http://127.0.0.1:5173')
  else window.loadFile(path.join(__dirname, '..', 'dist', 'index.html'))
}

app.whenReady().then(async () => {
  registerMediaProtocol()
  registerIpcHandlers()
  await ensureDefaultDemos()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
