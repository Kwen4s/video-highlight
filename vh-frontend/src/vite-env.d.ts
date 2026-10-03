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
  adStudio?: {
    importAsset: <T>(file: File, kind: 'video' | 'image') => Promise<T>
    listAssets: <T>() => Promise<T[]>
    deleteAsset: (assetId: string) => Promise<void>
    exportHighlight: <T>(input: unknown) => Promise<T>
    exportCleanHighlight: <T>(input: unknown) => Promise<T>
  }
}
