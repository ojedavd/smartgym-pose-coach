export const MAX_REFERENCE_BYTES = 100 * 1024 * 1024;
export const MAX_REFERENCE_SECONDS = 60;
const SAMPLE_RATE = 8;

export function validateReferenceFile(file: Pick<File, 'name' | 'type' | 'size'>) {
  if (file.size === 0) throw new Error('El archivo está vacío. Elegí otro video.');
  if (file.size > MAX_REFERENCE_BYTES) throw new Error('El video supera los 100 MB. Recortalo o exportalo con menor resolución.');
  const unknownType = !file.type || file.type === 'application/octet-stream';
  if (!file.type.startsWith('video/') && !(unknownType && /\.(mp4|webm|mov|m4v|ogv)$/i.test(file.name))) {
    throw new Error('Elegí un archivo de video, por ejemplo MP4, WebM o MOV.');
  }
}

export function validateReferenceDuration(duration: number) {
  if (!Number.isFinite(duration) || duration < 1) {
    throw new Error('No pude leer la duración del video. Probá exportarlo como MP4 y cargalo otra vez.');
  }
  if (duration > MAX_REFERENCE_SECONDS) {
    throw new Error('El video debe durar hasta 60 segundos. Recortalo para incluir las repeticiones de referencia.');
  }
}

function waitForMedia(
  video: HTMLVideoElement,
  event: 'loadeddata' | 'seeked',
  signal: AbortSignal,
  action: () => void,
) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timeout);
      video.removeEventListener(event, onReady);
      video.removeEventListener('error', onError);
      signal.removeEventListener('abort', onAbort);
    };
    const onReady = () => { cleanup(); resolve(); };
    const onError = () => {
      cleanup();
      reject(new Error('El navegador no puede reproducir este video. Probá un MP4 con códec H.264.'));
    };
    const onAbort = () => { cleanup(); reject(new DOMException('Carga cancelada', 'AbortError')); };
    const timeout = setTimeout(() => {
      cleanup();
      reject(new Error('El video tardó demasiado en abrirse. Probá con un archivo más corto o en formato MP4.'));
    }, 20000);
    video.addEventListener(event, onReady, { once: true });
    video.addEventListener('error', onError, { once: true });
    signal.addEventListener('abort', onAbort, { once: true });
    try { action(); } catch (error) { cleanup(); reject(error); }
  });
}

/** Decode local frames without playing audio, accessing the camera, or sending the file. */
export async function analyzeReferenceVideo(
  file: File,
  options: {
    signal: AbortSignal;
    onFrame: (video: HTMLVideoElement, timeMs: number) => void;
    onProgress: (percent: number) => void;
  },
) {
  validateReferenceFile(file);
  options.signal.throwIfAborted();
  const video = document.createElement('video');
  const url = URL.createObjectURL(file);
  video.preload = 'auto';
  video.muted = true;
  video.playsInline = true;
  try {
    await waitForMedia(video, 'loadeddata', options.signal, () => {
      video.src = url;
      video.load();
    });
    // Some MediaRecorder WebM files reveal their duration only after a seek.
    if (video.duration === Infinity) {
      await waitForMedia(video, 'seeked', options.signal, () => { video.currentTime = 1e10; });
    }
    const duration = video.duration;
    validateReferenceDuration(duration);
    if (!video.videoWidth || !video.videoHeight) throw new Error('El archivo no contiene imágenes de video.');
    const samples = Math.ceil(duration * SAMPLE_RATE);
    options.onProgress(0);
    for (let index = 0; index < samples; index += 1) {
      options.signal.throwIfAborted();
      const seconds = index / SAMPLE_RATE;
      if (Math.abs(video.currentTime - seconds) > 0.0001) {
        await waitForMedia(video, 'seeked', options.signal, () => { video.currentTime = seconds; });
      }
      options.signal.throwIfAborted();
      options.onFrame(video, seconds * 1000);
      options.onProgress(Math.round(((index + 1) / samples) * 100));
    }
    return duration;
  } finally {
    video.pause();
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}
