/// <reference types="vite/client" />

interface Window {
  desktopWindow?: {
    minimize: () => void
    toggleMaximize: () => void
    close: () => void
  }
}
