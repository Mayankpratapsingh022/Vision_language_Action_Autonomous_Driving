import { describe, expect, it } from 'vitest';
import type { DatasetExport } from '../types';
import {
  humanEpisodeFilename,
  saveDataset,
  type DatasetDirectoryHandle,
} from '../vla/datasetStorage';

describe('humanEpisodeFilename', () => {
  it('creates stable, sortable names for automatically collected episodes', () => {
    const filename = humanEpisodeFilename(
      'Turn Left / Intersection',
      42.4,
      7,
      new Date('2026-07-14T08:09:10.123Z'),
    );

    expect(filename).toBe(
      'human-turn_left_intersection-seed-42-run-0007-2026-07-14T08-09-10-123Z.json',
    );
  });

  it('writes an episode into the selected directory without another prompt', async () => {
    let savedName = '';
    const savedBlobs: Blob[] = [];
    const directory: DatasetDirectoryHandle = {
      async getFileHandle(name) {
        savedName = name;
        return {
          async createWritable() {
            return {
              async write(data) {
                savedBlobs.push(data);
              },
              async close() {},
            };
          },
        };
      },
    };
    const dataset: DatasetExport = {
      metadata: {
        image_width: 128,
        image_height: 128,
        frame_stack: 1,
        num_intents: 9,
        intent_labels: [],
        intent_texts: [],
        num_samples: 0,
        capture_rate_ms: 90,
        capture_resolution: 128,
        observation_keys: [],
        schema_version: 'vla-urban-3',
        created: '2026-07-14T08:09:10.123Z',
      },
      samples: [],
    };

    await expect(saveDataset(dataset, 'episode-1', directory)).resolves.toBe('directory');
    expect(savedName).toBe('episode-1.json');
    expect(savedBlobs).toHaveLength(1);
    expect(JSON.parse(await savedBlobs[0].text())).toEqual(dataset);
  });
});
