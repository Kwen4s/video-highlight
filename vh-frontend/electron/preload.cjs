const { contextBridge, ipcRenderer, webUtils } = require('electron')

contextBridge.exposeInMainWorld('desktopWindow', {
  minimize: () => ipcRenderer.send('window:minimize'),
  toggleMaximize: () => ipcRenderer.send('window:toggle-maximize'),
  close: () => ipcRenderer.send('window:close'),
})

contextBridge.exposeInMainWorld('adStudio', {
  importAsset: (file, kind) => {
    const sourcePath = webUtils.getPathForFile(file)
    if (!sourcePath) return Promise.reject(new Error('无法读取所选广告素材的本地路径'))
    return ipcRenderer.invoke('ads:import-asset', {
      sourcePath,
      originalName: file.name,
      kind,
    })
  },
  listAssets: () => ipcRenderer.invoke('ads:list-assets'),
  deleteAsset: (assetId) => ipcRenderer.invoke('ads:delete-asset', assetId),
  exportHighlight: (input) => ipcRenderer.invoke('ads:export-highlight', input),
  exportCleanHighlight: (input) => ipcRenderer.invoke('highlights:export-clean', input),
})
