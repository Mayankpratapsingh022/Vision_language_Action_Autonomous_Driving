import type { DatasetExport } from '../types';

interface DatasetWritableFile {
  write(data: Blob): Promise<void>;
  close(): Promise<void>;
}

interface DatasetFileHandle {
  createWritable(): Promise<DatasetWritableFile>;
}

export interface DatasetDirectoryHandle {
  getFileHandle(name: string, options: { create: true }): Promise<DatasetFileHandle>;
}

type DirectoryPickerWindow = Window & {
  showDirectoryPicker?: (options?: {
    id?: string;
    mode?: 'read' | 'readwrite';
    startIn?: 'desktop' | 'documents' | 'downloads';
  }) => Promise<DatasetDirectoryHandle>;
};

export async function chooseDatasetDirectory(): Promise<DatasetDirectoryHandle | null> {
  const picker = (window as DirectoryPickerWindow).showDirectoryPicker;
  if (!picker) return null;
  return picker.call(window, { id: 'vla-human-left-turn-data', mode: 'readwrite', startIn: 'downloads' });
}

export async function saveDataset(
  dataset: DatasetExport,
  filename: string,
  directory: DatasetDirectoryHandle | null,
): Promise<'directory' | 'download'> {
  const safeFilename = ensureJsonExtension(filename);
  const blob = new Blob([JSON.stringify(dataset)], { type: 'application/json' });
  if (directory) {
    const file = await directory.getFileHandle(safeFilename, { create: true });
    const writable = await file.createWritable();
    await writable.write(blob);
    await writable.close();
    return 'directory';
  }

  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = safeFilename;
  link.hidden = true;
  document.body.append(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1_000);
  return 'download';
}

export function humanEpisodeFilename(
  intentId: string,
  seed: number,
  episodeNumber: number,
  date = new Date(),
): string {
  const intent = intentId.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '');
  const timestamp = date.toISOString().replace(/[:.]/g, '-');
  return `human-${intent}-seed-${Math.max(0, Math.round(seed))}-run-${String(episodeNumber).padStart(4, '0')}-${timestamp}.json`;
}

function ensureJsonExtension(filename: string): string {
  return filename.toLowerCase().endsWith('.json') ? filename : `${filename}.json`;
}
