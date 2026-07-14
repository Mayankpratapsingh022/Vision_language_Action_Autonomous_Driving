import { spawnSync } from 'node:child_process';
import { mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';

const url = new URL(process.env.SIM_URL ?? 'http://127.0.0.1:4173/');
url.searchParams.set('signals', 'green');
const outputDirectory = resolve(process.env.OUTPUT_DIR ?? '/private/tmp/vla-driving-readme-capture');
const outputVideo = resolve(outputDirectory, 'driving-turn.webm');
const outputGif = resolve(process.env.GIF_OUTPUT ?? 'docs/images/vla-driving-observation-actions.gif');
await mkdir(outputDirectory, { recursive: true });
await mkdir(dirname(outputGif), { recursive: true });

const browser = await chromium.launch({ channel: 'chrome', headless: true });
const context = await browser.newContext({
  viewport: { width: 960, height: 540 },
  deviceScaleFactor: 1,
  recordVideo: { dir: outputDirectory, size: { width: 960, height: 540 } },
});
const page = await context.newPage();
const pageErrors = [];
let summary;
page.on('pageerror', (error) => pageErrors.push(error.message));

try {
  await page.goto(url.href, { waitUntil: 'networkidle' });
  await page.waitForSelector('#viewport');
  await page.waitForTimeout(2_500);
  await page.selectOption('#intent-select', 'turn_left_intersection');
  await page.selectOption('#density-select', 'low');
  await page.selectOption('#weather-select', 'clear');
  await page.selectOption('#quality-select', 'high');
  await page.click('[data-camera="chase"]');
  await page.waitForTimeout(1_000);
  await page.click('#expert-button');
  await page.addStyleTag({
    content: `
      #hud-root > * { display: none !important; }
      #hud-root > .action-panel {
        display: block !important;
        position: fixed !important;
        inset: auto 18px 18px auto !important;
        width: 320px !important;
        padding: 14px !important;
        z-index: 30 !important;
      }
      #hud-root > .action-panel .action-head span,
      #hud-root > .action-panel .action-head strong,
      #hud-root > .action-panel .action-row span,
      #hud-root > .action-panel .action-row strong {
        font-size: 11px !important;
      }
      #hud-root > .action-panel .action-track { height: 8px !important; }
      #hud-root > .action-panel .action-spark { height: 72px !important; }
      #sensor-preview {
        display: block !important;
        position: fixed !important;
        inset: auto auto 18px 18px !important;
        width: 190px !important;
        height: 190px !important;
        z-index: 30 !important;
      }
      #model-view-label {
        display: flex !important;
        position: fixed !important;
        inset: auto auto 216px 18px !important;
        width: 230px !important;
        justify-content: space-between !important;
        gap: 10px !important;
        white-space: nowrap !important;
        z-index: 31 !important;
      }
      #model-view-label span { letter-spacing: 0.05em !important; }
      #model-view-label strong {
        color: #243342 !important;
        font-size: 8px !important;
        letter-spacing: 0 !important;
      }
    `,
  });
  await page.evaluate(() => {
    const label = document.querySelector('#model-view-label');
    if (label) label.innerHTML = '<span>MODEL OBSERVATION</span><strong>128 x 128 PIXELS</strong>';
  });
  await page.waitForTimeout(22_000);

  const progress = await page.textContent('[data-status="progress"]');
  const collisions = await page.textContent('[data-status="collisions"]');
  await page.screenshot({ path: resolve(outputDirectory, 'final-frame.png') });
  if (pageErrors.length) throw new Error(pageErrors.join('\n'));
  if (Number(collisions) > 0) throw new Error(`Capture had ${collisions} collisions`);
  summary = { progress, collisions, outputVideo, outputGif };
} finally {
  const video = page.video();
  await context.close();
  if (video) await video.saveAs(outputVideo);
  await browser.close();
}

const encoding = spawnSync(
  'ffmpeg',
  [
    '-y',
    '-ss',
    '5',
    '-t',
    '14',
    '-i',
    outputVideo,
    '-filter_complex',
    'fps=8,scale=720:-1:flags=lanczos,split[frames][palette_source];[palette_source]palettegen=max_colors=80:stats_mode=diff[palette];[frames][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
    '-loop',
    '0',
    outputGif,
  ],
  { stdio: 'inherit' },
);
if (encoding.error) throw encoding.error;
if (encoding.status !== 0) throw new Error(`FFmpeg exited with status ${encoding.status}`);
console.log(JSON.stringify(summary));
