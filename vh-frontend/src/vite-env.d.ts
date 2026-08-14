/// <reference types="vite/client" />

interface Window {
  desktopWindow?: {
    minimize: () => void
    toggleMaximize: () => void
    close: () => void
  }
  videoImports?: {
    prepare: (input: { jobId: string; fileName: string }) => Promise<{
      jobId: string
      relativeDirectory: string
    }>
  }
}
