const { app, BrowserWindow, ipcMain } = require('electron')
const { mkdir } = require('node:fs/promises')
const path = require('node:path')

const isDevelopment = process.argv.includes('--dev')
const JOB_ID_PATTERN = /^job_[A-Za-z0-9_-]{8,48}$/

function getInstallRoot() {
  return app.isPackaged ? path.dirname(app.getPath('exe')) : path.resolve(__dirname, '..')
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

  ipcMain.handle('video-imports:prepare', async (_event, input) => {
    const jobId = typeof input?.jobId === 'string' ? input.jobId : ''
    if (!JOB_ID_PATTERN.test(jobId)) throw new Error('无效的任务编号')

    const jobDirectory = path.join(getInstallRoot(), 'video-data', 'jobs', jobId)
    await mkdir(path.join(jobDirectory, 'source'), { recursive: true })
    return { jobId, relativeDirectory: path.join('video-data', 'jobs', jobId) }
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

app.whenReady().then(() => {
  registerIpcHandlers()
  createWindow()
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})
