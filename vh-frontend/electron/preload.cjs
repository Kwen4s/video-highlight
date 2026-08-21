const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('desktopWindow', {
  minimize: () => ipcRenderer.send('window:minimize'),
  toggleMaximize: () => ipcRenderer.send('window:toggle-maximize'),
  close: () => ipcRenderer.send('window:close'),
})

contextBridge.exposeInMainWorld('localLibrary', {
  importSource: (file, input) => {
    const sourcePath = webUtils.getPathForFile(file)
    if (!sourcePath) return Promise.reject(new Error('无法读取所选文件的本地路径'))
    return ipcRenderer.invoke('library:import-source', { ...input, sourcePath })
  },
  saveJob: (job) => ipcRenderer.invoke('library:save-job', job),
  listJobs: () => ipcRenderer.invoke('library:list-jobs'),
  deleteJob: (jobId) => ipcRenderer.invoke('library:delete-job', jobId),
})
