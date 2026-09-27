import assert from 'node:assert/strict';
import { afterEach, mock, test } from 'node:test';
import {
  analyzeReferenceVideo,
  MAX_REFERENCE_BYTES,
  validateReferenceDuration,
  validateReferenceFile,
} from '../lib/reference-upload.ts';

const file = { name: 'sentadilla.mp4', type: 'video/mp4', size: 1024 };
const originalDocument = globalThis.document;

afterEach(() => {
  mock.restoreAll();
  if (originalDocument === undefined) delete globalThis.document;
  else globalThis.document = originalDocument;
});

function mockVideo({ duration = 2, decodeError = false, stalled = false } = {}) {
  class Video extends EventTarget {
    duration = duration;
    videoWidth = 1280;
    videoHeight = 720;
    time = 0;
    paused = false;
    cleaned = false;

    get currentTime() { return this.time; }
    set currentTime(value) {
      if (value === 1e10) { this.duration = 2; value = 2; }
      this.time = value;
      queueMicrotask(() => this.dispatchEvent(new Event('seeked')));
    }

    load() {
      if (this.cleaned || stalled) return;
      queueMicrotask(() => this.dispatchEvent(new Event(decodeError ? 'error' : 'loadeddata')));
    }
    pause() { this.paused = true; }
    removeAttribute(name) { if (name === 'src') this.cleaned = true; }
  }
  const video = new Video();
  Object.defineProperty(globalThis, 'document', {
    configurable: true, writable: true, value: { createElement: () => video },
  });
  mock.method(URL, 'createObjectURL', () => 'blob:test-reference');
  const revoke = mock.method(URL, 'revokeObjectURL', () => {});
  return { video, revoke };
}

test('validates type and size, including camera files without a MIME type', () => {
  assert.doesNotThrow(() => validateReferenceFile(file));
  assert.doesNotThrow(() => validateReferenceFile({ ...file, type: '', name: 'MOVIMIENTO.MOV' }));
  assert.doesNotThrow(() => validateReferenceFile({ ...file, size: MAX_REFERENCE_BYTES }));
  assert.throws(() => validateReferenceFile({ ...file, size: 0 }), /vacío/);
  assert.throws(() => validateReferenceFile({ ...file, size: MAX_REFERENCE_BYTES + 1 }), /100 MB/);
  assert.throws(() => validateReferenceFile({ ...file, name: 'imagen.png', type: 'image/png' }), /archivo de video/);
  assert.throws(() => validateReferenceFile({ ...file, name: 'documento.pdf', type: '' }), /archivo de video/);
});

test('accepts one to sixty seconds and rejects unreadable or excessive durations', () => {
  assert.doesNotThrow(() => validateReferenceDuration(1));
  assert.doesNotThrow(() => validateReferenceDuration(60));
  for (const duration of [0, 0.5, NaN, Infinity]) {
    assert.throws(() => validateReferenceDuration(duration), /duración/);
  }
  assert.throws(() => validateReferenceDuration(60.1), /60 segundos/);
});

test('samples the full local video at eight frames per second and releases resources', async () => {
  const { video, revoke } = mockVideo();
  const timestamps = [];
  const progress = [];
  const duration = await analyzeReferenceVideo(file, {
    signal: new AbortController().signal,
    onFrame: (source, time) => { assert.equal(source, video); timestamps.push(time); },
    onProgress: (value) => progress.push(value),
  });
  assert.equal(duration, 2);
  assert.deepEqual(timestamps, Array.from({ length: 16 }, (_, index) => index * 125));
  assert.equal(progress[0], 0);
  assert.equal(progress.at(-1), 100);
  assert.equal(video.muted, true);
  assert.equal(video.paused, true);
  assert.equal(video.cleaned, true);
  assert.equal(revoke.mock.callCount(), 1);
});

test('resolves the duration of MediaRecorder WebM files before sampling', async () => {
  mockVideo({ duration: Infinity });
  const timestamps = [];
  assert.equal(await analyzeReferenceVideo(file, {
    signal: new AbortController().signal,
    onFrame: (_, time) => timestamps.push(time),
    onProgress: () => {},
  }), 2);
  assert.equal(timestamps[0], 0);
  assert.equal(timestamps.length, 16);
});

test('cancels during analysis and cleans the decoder', async () => {
  const { video, revoke } = mockVideo();
  const controller = new AbortController();
  let frames = 0;
  await assert.rejects(analyzeReferenceVideo(file, {
    signal: controller.signal,
    onFrame: () => { frames += 1; controller.abort(); },
    onProgress: () => {},
  }), { name: 'AbortError' });
  assert.equal(frames, 1);
  assert.equal(video.cleaned, true);
  assert.equal(revoke.mock.callCount(), 1);
});

test('cancels while waiting for media to open', async () => {
  const { revoke } = mockVideo({ stalled: true });
  const controller = new AbortController();
  const analysis = analyzeReferenceVideo(file, {
    signal: controller.signal, onFrame: () => {}, onProgress: () => {},
  });
  controller.abort();
  await assert.rejects(analysis, { name: 'AbortError' });
  assert.equal(revoke.mock.callCount(), 1);
});

test('cleans up unsupported codecs and propagates the useful error', async () => {
  const { video, revoke } = mockVideo({ decodeError: true });
  await assert.rejects(analyzeReferenceVideo(file, {
    signal: new AbortController().signal, onFrame: () => {}, onProgress: () => {},
  }), /H.264/);
  assert.equal(video.cleaned, true);
  assert.equal(revoke.mock.callCount(), 1);
});

test('cleans up if pose inference fails', async () => {
  const { revoke } = mockVideo();
  await assert.rejects(analyzeReferenceVideo(file, {
    signal: new AbortController().signal,
    onFrame: () => { throw new Error('inference failed'); },
    onProgress: () => {},
  }), /inference failed/);
  assert.equal(revoke.mock.callCount(), 1);
});
