const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('desktopWindow', {
  minimize: () => ipcRenderer.send('window:minimize'),
  toggleMaximize: () => ipcRenderer.send('window:toggle-maximize'),
  close: () => ipcRenderer.send('window:close'),
})

contextBridge.exposeInMainWorld('videoImports', {
  prepare: (input) => ipcRenderer.invoke('video-imports:prepare', input),
})
