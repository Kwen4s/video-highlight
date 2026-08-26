/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}

interface Window {
  desktopWindow?: {
    minimize: () => void
    toggleMaximize: () => void
    close: () => void
  }
  localLibrary?: {
    importSource: <T>(file: File, input: {
      jobId: string
      originalName: string
      contentType: string
      language: 'zh' | 'en'
    }) => Promise<T>
    saveJob: <T>(job: T) => Promise<T>
    listJobs: <T>() => Promise<T[]>
    deleteJob: (jobId: string) => Promise<void>
  }
  adStudio?: {
    importAsset: <T>(file: File, kind: 'video' | 'image') => Promise<T>
    listAssets: <T>() => Promise<T[]>
    deleteAsset: (assetId: string) => Promise<void>
    exportHighlight: <T>(input: unknown) => Promise<T>
  }
}
