'use client';

import type { NormalizedLandmark, PoseLandmarker } from '@mediapipe/tasks-vision';
import {
  Activity,
  AlertTriangle,
  Camera,
  Check,
  CircleHelp,
  Download,
  LoaderCircle,
  Maximize2,
  Minimize2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  ShieldCheck,
  Square,
  Upload,
  Video,
  Volume2,
  VolumeX,
} from 'lucide-react';
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from 'react';

import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet';
import { Switch } from '@/components/ui/switch';
import { loadDemo, saveDemo } from '@/lib/demo-store';
import { analyzeReferenceVideo, validateReferenceFile } from '@/lib/reference-upload';
import {
  assessPose,
  buildReferenceProfile,
  calculatePoseMetrics,
  drawPose,
  formatDuration,
  INITIAL_REP_STATE,
  updateRepState,
  type PoseAssessment,
  type ReferenceFrame,
  type ReferenceProfile,
  type RepState,
  type TrainingLevel,
} from '@/lib/pose-analysis';

type ModelStatus = 'idle' | 'loading' | 'ready' | 'error';

interface InstallPromptEvent extends Event {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

interface WebMCPContext {
  registerTool: (
    tool: {
      name: string;
      title: string;
      description: string;
      inputSchema: Record<string, unknown>;
      annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
      execute: (input: unknown) => unknown;
    },
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
}

const IDLE_ASSESSMENT: PoseAssessment = {
  segments: { trunk: 'neutral', thigh: 'neutral', shin: 'neutral' },
  primary: {
    region: 'general',
    title: 'Listo para analizar tu movimiento',
    detail: 'Grabá o subí una demostración y luego activá la cámara del alumno.',
    severity: 'success',
  },
  items: [
    { label: 'Espalda', value: 'En espera', state: 'neutral' },
    { label: 'Cadera', value: 'En espera', state: 'neutral' },
    { label: 'Rodilla', value: 'En espera', state: 'neutral' },
  ],
  improper: false,
  aligned: false,
  state: null,
};

const GOOD_SEGMENTS = { trunk: 'good', thigh: 'good', shin: 'good' } as const;
const MAX_RECORDING_SECONDS = 12;
// Ask for a taller frame where available; never crop it in the presentation layer.
const CAMERA_VIDEO_CONSTRAINTS: MediaTrackConstraints = {
  facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 960 }, aspectRatio: { ideal: 4 / 3 },
};
const subscribeToVoiceSupport = () => () => {};
const getVoiceSupport = () => 'speechSynthesis' in window && 'SpeechSynthesisUtterance' in window;
const getServerVoiceSupport = () => false;

export function Trainer() {
  const instructorVideo = useRef<HTMLVideoElement>(null);
  const instructorCanvas = useRef<HTMLCanvasElement>(null);
  const studentVideo = useRef<HTMLVideoElement>(null);
  const studentCanvas = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<PoseLandmarker | null>(null);
  const modelPromiseRef = useRef<Promise<PoseLandmarker> | null>(null);
  const instructorStreamRef = useRef<MediaStream | null>(null);
  const studentStreamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const referenceFramesRef = useRef<ReferenceFrame[]>([]);
  const recordingStartRef = useRef(0);
  const recordingRafRef = useRef(0);
  const practiceRafRef = useRef(0);
  const autoStopRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const repRef = useRef<RepState>({ ...INITIAL_REP_STATE, sequence: [] });
  const lastUiUpdateRef = useRef(0);
  const fpsRef = useRef({ frames: 0, startedAt: 0 });
  const demoUrlRef = useRef<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadAbortRef = useRef<AbortController | null>(null);
  const referenceBusyRef = useRef(false);
  const practiceStartingRef = useRef(false);
  const spokenRef = useRef({ text: '', at: 0 });
  const controlsTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [level, setLevel] = useState<TrainingLevel>('beginner');
  const [modelStatus, setModelStatus] = useState<ModelStatus>('idle');
  const [referenceProfile, setReferenceProfile] = useState<ReferenceProfile | null>(null);
  const [demoUrl, setDemoUrl] = useState<string | null>(null);
  const [referenceBusy, setReferenceBusy] = useState(false);
  const [practiceStarting, setPracticeStarting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [recording, setRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [practicing, setPracticing] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [controlsVisible, setControlsVisible] = useState(true);
  const [splitView, setSplitView] = useState(false);
  const [referenceFocus, setReferenceFocus] = useState(false);
  const [showInstructorPose, setShowInstructorPose] = useState(true);
  const [referenceSheetOpen, setReferenceSheetOpen] = useState(false);
  const [settingsSheetOpen, setSettingsSheetOpen] = useState(false);
  const voiceSupported = useSyncExternalStore(subscribeToVoiceSupport, getVoiceSupport, getServerVoiceSupport);
  const [assessment, setAssessment] = useState<PoseAssessment>(IDLE_ASSESSMENT);
  const [repCounts, setRepCounts] = useState({ correct: 0, incorrect: 0 });
  const [fps, setFps] = useState(0);
  const [notice, setNotice] = useState<string | null>(null);
  const [installPrompt, setInstallPrompt] = useState<InstallPromptEvent | null>(null);

  const setDemoObjectUrl = useCallback((blob: Blob) => {
    if (demoUrlRef.current) URL.revokeObjectURL(demoUrlRef.current);
    const nextUrl = URL.createObjectURL(blob);
    demoUrlRef.current = nextUrl;
    setDemoUrl(nextUrl);
    return nextUrl;
  }, []);

  useEffect(() => {
    let active = true;
    void loadDemo()
      .then((stored) => {
        if (!active || !stored) return;
        setReferenceProfile(stored.profile);
        setDemoObjectUrl(stored.blob);
      })
      .catch(() => setNotice('No se pudo recuperar la demostración guardada. Podés grabar una nueva.'));

    if ('serviceWorker' in navigator) {
      void navigator.serviceWorker.register('/sw.js').catch(() => undefined);
    }
    const captureInstallPrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as InstallPromptEvent);
    };
    window.addEventListener('beforeinstallprompt', captureInstallPrompt);

    return () => {
      active = false;
      uploadAbortRef.current?.abort();
      window.removeEventListener('beforeinstallprompt', captureInstallPrompt);
      cancelAnimationFrame(recordingRafRef.current);
      cancelAnimationFrame(practiceRafRef.current);
      if (controlsTimerRef.current) clearTimeout(controlsTimerRef.current);
      instructorStreamRef.current?.getTracks().forEach((track) => track.stop());
      studentStreamRef.current?.getTracks().forEach((track) => track.stop());
      landmarkerRef.current?.close();
      if (demoUrlRef.current) URL.revokeObjectURL(demoUrlRef.current);
    };
  }, [setDemoObjectUrl]);

  useEffect(() => {
    if (!voiceEnabled || !practicing || !voiceSupported) return;
    const text = assessment.primary.title;
    // Wait for a stable instruction, avoid a spoken queue, and space corrections apart.
    const delay = Math.max(800, 4000 - (Date.now() - spokenRef.current.at));
    const timer = window.setTimeout(() => {
      if (spokenRef.current.text === text) return;
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'es-AR';
      utterance.rate = 0.95;
      utterance.onerror = (event) => {
        if (event.error !== 'canceled' && event.error !== 'interrupted') {
          setNotice('La voz no está disponible en este navegador. Podés seguir con el feedback grande.');
          setVoiceEnabled(false);
        }
      };
      window.speechSynthesis.speak(utterance);
      spokenRef.current = { text, at: Date.now() };
    }, delay);
    return () => window.clearTimeout(timer);
  }, [voiceEnabled, voiceSupported, practicing, assessment.primary.title]);

  useEffect(() => {
    if (!voiceSupported) return;
    if (!voiceEnabled || !practicing) {
      window.speechSynthesis.cancel();
      spokenRef.current = { text: '', at: 0 };
    }
    return () => window.speechSynthesis.cancel();
  }, [voiceEnabled, voiceSupported, practicing]);

  const revealControls = useCallback(() => {
    setControlsVisible(true);
    if (controlsTimerRef.current) clearTimeout(controlsTimerRef.current);
    if (practicing && !referenceSheetOpen && !settingsSheetOpen) {
      controlsTimerRef.current = setTimeout(() => setControlsVisible(false), 3500);
    }
  }, [practicing, referenceSheetOpen, settingsSheetOpen]);

  useEffect(() => {
    if (!practicing || referenceSheetOpen || settingsSheetOpen) {
      setControlsVisible(true);
      if (controlsTimerRef.current) clearTimeout(controlsTimerRef.current);
      return;
    }
    revealControls();
    return () => {
      if (controlsTimerRef.current) clearTimeout(controlsTimerRef.current);
    };
  }, [practicing, referenceSheetOpen, settingsSheetOpen, revealControls]);

  useEffect(() => {
    const context = (document as Document & { modelContext?: WebMCPContext }).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const registration = context.registerTool(
      {
        name: 'configure_training_level',
        title: 'Configurar nivel de entrenamiento',
        description: 'Cambia el nivel visible de tolerancia postural entre principiante y avanzado.',
        inputSchema: {
          type: 'object',
          properties: { level: { type: 'string', enum: ['beginner', 'pro'] } },
          required: ['level'],
          additionalProperties: false,
        },
        annotations: { readOnlyHint: false, untrustedContentHint: false },
        execute(input) {
          const nextLevel = (input as { level?: unknown })?.level;
          if (nextLevel !== 'beginner' && nextLevel !== 'pro') {
            throw new Error('El nivel debe ser beginner o pro.');
          }
          setLevel(nextLevel);
          return { level: nextLevel, status: 'configured' };
        },
      },
      { signal: lifecycle.signal },
    );
    void Promise.resolve(registration).catch(() => undefined);
    return () => lifecycle.abort();
  }, []);

  useEffect(() => {
    if (!recording) return;
    const interval = window.setInterval(() => {
      setRecordingSeconds(Math.min(MAX_RECORDING_SECONDS, (performance.now() - recordingStartRef.current) / 1000));
    }, 100);
    return () => window.clearInterval(interval);
  }, [recording]);

  useEffect(() => {
    if (!demoUrl || !referenceProfile || recording) return;
    let animationFrame = 0;
    const renderReference = () => {
      const video = instructorVideo.current;
      const canvas = instructorCanvas.current;
      if (video && canvas && video.readyState >= 2 && referenceProfile.frames.length) {
        syncCanvas(canvas, video);
        if (showInstructorPose) {
          const time = video.currentTime * 1000;
          const frame = referenceProfile.frames.reduce((closest, candidate) => (
            Math.abs(candidate.t - time) < Math.abs(closest.t - time) ? candidate : closest
          ));
          drawPose(canvas, frame.landmarks, GOOD_SEGMENTS, true);
        } else {
          clearCanvas(canvas);
        }
      }
      animationFrame = requestAnimationFrame(renderReference);
    };
    animationFrame = requestAnimationFrame(renderReference);
    return () => cancelAnimationFrame(animationFrame);
  }, [demoUrl, recording, referenceProfile, showInstructorPose]);

  const ensureLandmarker = useCallback(async () => {
    if (landmarkerRef.current) return landmarkerRef.current;
    if (modelPromiseRef.current) return modelPromiseRef.current;
    setModelStatus('loading');
    modelPromiseRef.current = (async () => {
      const { FilesetResolver, PoseLandmarker: PoseLandmarkerClass } = await import('@mediapipe/tasks-vision');
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm',
      );
      const options = {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU' as const,
        },
        runningMode: 'VIDEO' as const,
        numPoses: 1,
        minPoseDetectionConfidence: 0.55,
        minPosePresenceConfidence: 0.55,
        minTrackingConfidence: 0.55,
        outputSegmentationMasks: false,
      };
      let landmarker: PoseLandmarker;
      try {
        landmarker = await PoseLandmarkerClass.createFromOptions(vision, options);
      } catch {
        landmarker = await PoseLandmarkerClass.createFromOptions(vision, {
          ...options,
          baseOptions: { ...options.baseOptions, delegate: 'CPU' },
        });
      }
      landmarkerRef.current = landmarker;
      setModelStatus('ready');
      return landmarker;
    })().catch((error) => {
      modelPromiseRef.current = null;
      setModelStatus('error');
      throw error;
    });
    return modelPromiseRef.current;
  }, []);

  const stopReferenceRecording = useCallback(() => {
    if (autoStopRef.current) clearTimeout(autoStopRef.current);
    cancelAnimationFrame(recordingRafRef.current);
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop();
    setRecording(false);
  }, []);

  const stopPractice = useCallback(() => {
    cancelAnimationFrame(practiceRafRef.current);
    studentStreamRef.current?.getTracks().forEach((track) => track.stop());
    studentStreamRef.current = null;
    if (studentVideo.current) studentVideo.current.srcObject = null;
    if (studentCanvas.current) clearCanvas(studentCanvas.current);
    setPracticing(false);
    setFps(0);
  }, []);

  const startReferenceRecording = async () => {
    if (recording) {
      stopReferenceRecording();
      return;
    }
    if (referenceBusyRef.current || practiceStartingRef.current) return;
    if (practicing) stopPractice();
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setNotice('Este navegador no permite grabar desde la cámara. Probá con Chrome, Edge o Safari actualizado.');
      return;
    }

    referenceBusyRef.current = true;
    setReferenceBusy(true);
    setReferenceFocus(true);
    setReferenceSheetOpen(false);
    try {
      setNotice('Preparando el análisis corporal…');
      const landmarker = await ensureLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: CAMERA_VIDEO_CONSTRAINTS,
        audio: false,
      });
      instructorStreamRef.current = stream;
      const video = instructorVideo.current;
      if (!video) throw new Error('La vista de cámara no está disponible.');
      video.pause();
      video.removeAttribute('src');
      video.srcObject = stream;
      video.muted = true;
      await video.play();

      const mimeType = preferredMimeType();
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      chunksRef.current = [];
      referenceFramesRef.current = [];
      recordingStartRef.current = performance.now();
      setRecordingSeconds(0);
      setRecording(true);
      setNotice('Hacé dos sentadillas completas y controladas. La grabación se detiene a los 12 segundos.');

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = async () => {
        try {
          const duration = (performance.now() - recordingStartRef.current) / 1000;
          const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'video/webm' });
          stream.getTracks().forEach((track) => track.stop());
          instructorStreamRef.current = null;
          if (referenceFramesRef.current.length < 5 || blob.size === 0) {
            setNotice('No pude detectar una demostración completa. Grabala otra vez con todo el cuerpo visible.');
            return;
          }
          const profile = buildReferenceProfile(referenceFramesRef.current, duration);
          const url = setDemoObjectUrl(blob);
          setReferenceProfile(profile);
          setNotice('Demostración guardada. Ya podés iniciar la práctica del alumno.');
          try {
            await saveDemo({ blob, profile });
          } catch {
            setNotice('La demostración funciona, pero el navegador no pudo guardarla para la próxima visita.');
          }
          video.srcObject = null;
          video.src = url;
          video.loop = true;
          video.controls = true;
          video.onloadedmetadata = () => void video.play().catch(() => undefined);
        } finally {
          setReferenceFocus(false);
          referenceBusyRef.current = false;
          setReferenceBusy(false);
        }
      };
      recorder.start(250);

      let lastInference = 0;
      const analyzeReference = (now: number) => {
        if (video.readyState >= 2 && now - lastInference >= 55) {
          lastInference = now;
          const result = landmarker.detectForVideo(video, now);
          const landmarks = result.landmarks[0];
          if (landmarks) {
            const metrics = calculatePoseMetrics(landmarks, video.videoWidth / video.videoHeight);
            if (metrics) {
              referenceFramesRef.current.push({
                t: now - recordingStartRef.current,
                metrics,
                landmarks: cloneLandmarks(landmarks),
              });
              const canvas = instructorCanvas.current;
              if (canvas) {
                syncCanvas(canvas, video);
                if (showInstructorPose) drawPose(canvas, landmarks, GOOD_SEGMENTS, true, metrics.side);
                else clearCanvas(canvas);
              }
            }
          }
        }
        if (recorder.state === 'recording') recordingRafRef.current = requestAnimationFrame(analyzeReference);
      };
      recordingRafRef.current = requestAnimationFrame(analyzeReference);
      autoStopRef.current = setTimeout(stopReferenceRecording, MAX_RECORDING_SECONDS * 1000);
    } catch (error) {
      referenceBusyRef.current = false;
      setReferenceBusy(false);
      setRecording(false);
      instructorStreamRef.current?.getTracks().forEach((track) => track.stop());
      instructorStreamRef.current = null;
      setReferenceFocus(false);
      setNotice(cameraErrorMessage(error));
    }
  };

  const uploadReference = async (file: File) => {
    if (referenceBusyRef.current || practicing || practiceStartingRef.current) return;
    try { validateReferenceFile(file); } catch (error) {
      setNotice(error instanceof Error ? error.message : 'No se pudo abrir el archivo.');
      return;
    }
    const controller = new AbortController();
    uploadAbortRef.current = controller;
    referenceBusyRef.current = true;
    setReferenceBusy(true);
    setUploading(true);
    setUploadProgress(0);
    setNotice('Analizando el video en este dispositivo. No cierres la página hasta que termine.');
    try {
      const landmarker = await ensureLandmarker();
      controller.signal.throwIfAborted();
      const frames: ReferenceFrame[] = [];
      const duration = await analyzeReferenceVideo(file, {
        signal: controller.signal,
        onProgress: setUploadProgress,
        onFrame(video, t) {
          const landmarks = landmarker.detectForVideo(video, performance.now()).landmarks[0];
          if (!landmarks) return;
          const metrics = calculatePoseMetrics(landmarks, video.videoWidth / video.videoHeight);
          if (metrics && metrics.visibility >= 0.45 && metrics.offsetAngle <= 38) {
            frames.push({ t, metrics, landmarks: cloneLandmarks(landmarks) });
          }
        },
      });
      controller.signal.throwIfAborted();
      if (frames.length < 5) {
        throw new Error('No encontré suficientes posturas de perfil. Elegí un video con todo el cuerpo visible y buena iluminación.');
      }
      const profile = buildReferenceProfile(frames, duration);
      let stored = true;
      try { await saveDemo({ blob: file, profile, fileName: file.name }); } catch { stored = false; }
      // Once persistence starts, finish the replacement atomically with the visible reference.
      instructorVideo.current?.pause();
      if (instructorVideo.current) instructorVideo.current.srcObject = null;
      setDemoObjectUrl(file);
      setReferenceProfile(profile);
      repRef.current = { ...INITIAL_REP_STATE, sequence: [] };
      setRepCounts({ correct: 0, incorrect: 0 });
      setAssessment(IDLE_ASSESSMENT);
      setNotice(stored
        ? 'Video analizado y guardado. Ya podés iniciar la práctica del alumno.'
        : 'Video listo para practicar. No hubo espacio para guardarlo: deberás cargarlo otra vez al volver.');
      setReferenceSheetOpen(false);
    } catch (error) {
      setNotice(error instanceof DOMException && error.name === 'AbortError'
        ? 'Carga cancelada. Se conservó la demostración anterior.'
        : `${error instanceof Error ? error.message : 'No pude analizar el video.'} Se conservó la demostración anterior.`);
    } finally {
      if (uploadAbortRef.current === controller) uploadAbortRef.current = null;
      referenceBusyRef.current = false;
      setReferenceBusy(false);
      setUploading(false);
    }
  };

  const startPractice = async () => {
    if (practicing) {
      stopPractice();
      return;
    }
    if (referenceBusyRef.current || practiceStartingRef.current) return;
    if (!referenceProfile) {
      setNotice('Primero grabá o subí la demostración del instructor. Así puedo comparar cada fase del movimiento.');
      setReferenceSheetOpen(true);
      return;
    }
    practiceStartingRef.current = true;
    setPracticeStarting(true);
    try {
      setNotice('Activando la cámara del alumno…');
      const landmarker = await ensureLandmarker();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: CAMERA_VIDEO_CONSTRAINTS,
        audio: false,
      });
      studentStreamRef.current = stream;
      const video = studentVideo.current;
      if (!video) return;
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      void instructorVideo.current?.play().catch(() => undefined);
      setPracticing(true);
      setReferenceFocus(false);
      setAssessment({ ...IDLE_ASSESSMENT, primary: { region: 'camera', title: 'Ponete de perfil', detail: 'Mostrá el cuerpo completo, desde el hombro hasta el pie.', severity: 'warning' } });
      setNotice('Análisis activo. Ponete de perfil y dejá todo el cuerpo dentro del cuadro.');
      fpsRef.current = { frames: 0, startedAt: performance.now() };

      let lastInference = 0;
      const analyzeStudent = (now: number) => {
        if (video.readyState >= 2 && now - lastInference >= 42) {
          lastInference = now;
          const result = landmarker.detectForVideo(video, now);
          const landmarks = result.landmarks[0];
          const canvas = studentCanvas.current;
          if (canvas) syncCanvas(canvas, video);
          if (landmarks && canvas) {
            const metrics = calculatePoseMetrics(landmarks, video.videoWidth / video.videoHeight);
            if (metrics) {
              const nextAssessment = assessPose(metrics, referenceProfile, level, repRef.current);
              const previousCounts = { correct: repRef.current.correct, incorrect: repRef.current.incorrect };
              repRef.current = updateRepState(nextAssessment.state, nextAssessment.improper, repRef.current);
              drawPose(canvas, landmarks, nextAssessment.segments, false, metrics.side);
              fpsRef.current.frames += 1;
              const elapsed = now - fpsRef.current.startedAt;
              if (elapsed >= 1000) {
                setFps(Math.round((fpsRef.current.frames * 1000) / elapsed));
                fpsRef.current = { frames: 0, startedAt: now };
              }
              if (
                now - lastUiUpdateRef.current > 120 ||
                previousCounts.correct !== repRef.current.correct ||
                previousCounts.incorrect !== repRef.current.incorrect
              ) {
                lastUiUpdateRef.current = now;
                setAssessment(nextAssessment);
                setRepCounts({ correct: repRef.current.correct, incorrect: repRef.current.incorrect });
              }
            }
          } else if (canvas) {
            clearCanvas(canvas);
            if (now - lastUiUpdateRef.current > 500) {
              lastUiUpdateRef.current = now;
              setAssessment({
                ...IDLE_ASSESSMENT,
                primary: {
                  region: 'camera',
                  title: 'No encuentro el cuerpo completo',
                  detail: 'Orientá la cámara para ver hombro y pie, y revisá la luz. No hace falta llenar el cuadro.',
                  severity: 'warning',
                },
              });
            }
          }
        }
        if (studentStreamRef.current) practiceRafRef.current = requestAnimationFrame(analyzeStudent);
      };
      practiceRafRef.current = requestAnimationFrame(analyzeStudent);
    } catch (error) {
      stopPractice();
      setNotice(cameraErrorMessage(error));
    } finally {
      practiceStartingRef.current = false;
      setPracticeStarting(false);
    }
  };

  const resetSession = () => {
    repRef.current = { ...INITIAL_REP_STATE, sequence: [] };
    setRepCounts({ correct: 0, incorrect: 0 });
    setAssessment(practicing ? { ...IDLE_ASSESSMENT, primary: { ...IDLE_ASSESSMENT.primary, title: 'Contadores reiniciados' } } : IDLE_ASSESSMENT);
    setNotice('Sesión reiniciada.');
  };

  const installApp = async () => {
    if (!installPrompt) {
      setNotice('Para instalarla, abrí el menú del navegador y elegí “Agregar a pantalla de inicio”.');
      return;
    }
    await installPrompt.prompt();
    const choice = await installPrompt.userChoice;
    if (choice.outcome === 'accepted') setNotice('Forma se instaló correctamente.');
    setInstallPrompt(null);
  };

  const primaryTone = assessment.primary.severity;
  const coachMessage = recording
    ? 'Grabando demostración'
    : uploading
      ? `Analizando ${uploadProgress}%`
      : practicing
        ? assessment.primary.title
        : referenceProfile
          ? 'Listo para entrenar'
          : 'Agregá una demostración';
  const coachTone = recording ? 'error' : uploading ? 'warning' : practicing ? primaryTone : 'success';

  return (
    <main className="coach-shell">
      <section
        className={`coach-stage ${splitView ? 'is-split' : ''} ${referenceFocus ? 'is-reference-focus' : ''}`}
        onPointerDown={revealControls}
        aria-label="Entrenamiento de sentadillas"
      >
        <div className={`coach-topbar ${controlsVisible ? 'is-visible' : ''}`}>
          <div className="coach-brand" aria-label="Forma Motion Coach">
            <span><Activity aria-hidden="true" /></span>
            <div>
              <strong>Forma</strong>
              <small>Sentadillas</small>
            </div>
          </div>
          <div className="coach-counts" aria-label={`${repCounts.correct} repeticiones correctas, ${repCounts.incorrect} a corregir`}>
            <span className="is-good"><Check aria-hidden="true" /> {String(repCounts.correct).padStart(2, '0')}</span>
            <span className="is-bad"><AlertTriangle aria-hidden="true" /> {String(repCounts.incorrect).padStart(2, '0')}</span>
          </div>
        </div>

        <div className="student-pane" aria-label="Cámara del alumno">
          <video ref={studentVideo} className="coach-video" playsInline muted autoPlay />
          <canvas ref={studentCanvas} className="coach-canvas" aria-hidden="true" />
          {!practicing && (
            <div className="coach-empty">
              <Camera aria-hidden="true" />
              <p>{referenceProfile ? 'Tocá play para empezar' : 'Agregá una demostración'}</p>
            </div>
          )}
          {modelStatus === 'loading' && !recording && <ModelLoading />}
          {practicing && <span className="live-chip"><span /> EN VIVO</span>}
        </div>

        <div
          className="reference-pane"
          role="button"
          tabIndex={0}
          aria-label={referenceFocus ? 'Volver a la cámara del alumno' : 'Ampliar demostración del instructor'}
          onClick={() => !splitView && setReferenceFocus((value) => !value)}
          onKeyDown={(event) => {
            if (!splitView && (event.key === 'Enter' || event.key === ' ')) {
              event.preventDefault();
              setReferenceFocus((value) => !value);
            }
          }}
        >
          <video
            ref={instructorVideo}
            src={demoUrl ?? undefined}
            className="coach-video"
            playsInline
            muted
            loop
            autoPlay={Boolean(demoUrl)}
            onLoadedMetadata={(event) => {
              if (recording || !referenceProfile) return;
              const video = event.currentTarget;
              const frames = referenceProfile.frames.flatMap((frame) => {
                const metrics = calculatePoseMetrics(frame.landmarks, video.videoWidth / video.videoHeight);
                return metrics ? [{ ...frame, metrics }] : [];
              });
              setReferenceProfile({ ...buildReferenceProfile(frames, referenceProfile.duration), createdAt: referenceProfile.createdAt });
            }}
          />
          <canvas ref={instructorCanvas} className="coach-canvas" aria-hidden="true" />
          {!recording && !demoUrl && <div className="reference-empty"><Video aria-hidden="true" /></div>}
          <span className="pane-label">Instructor</span>
          {recording && (
            <span className="recording-chip"><span /> REC {formatDuration(recordingSeconds)}</span>
          )}
          {uploading && (
            <div className="reference-loading" aria-live="polite">
              <LoaderCircle aria-hidden="true" />
              <span>{uploadProgress}%</span>
            </div>
          )}
        </div>

        <div className={`coach-feedback tone-${coachTone}`} aria-live="polite" aria-atomic="true">
          {coachTone === 'success' ? <Check aria-hidden="true" /> : <AlertTriangle aria-hidden="true" />}
          <strong>{coachMessage}</strong>
        </div>

        {notice && (
          <output className="coach-notice">
            <CircleHelp aria-hidden="true" />
            <span>{notice}</span>
            <button onClick={() => setNotice(null)} aria-label="Cerrar aviso">×</button>
          </output>
        )}

        <nav className={`coach-controls ${controlsVisible ? 'is-visible' : ''}`} aria-label="Controles de práctica">
          <button
            type="button"
            className={voiceEnabled ? 'is-active' : ''}
            aria-label={voiceEnabled ? 'Desactivar voz' : 'Activar voz'}
            aria-pressed={voiceEnabled}
            disabled={!voiceSupported}
            onClick={() => setVoiceEnabled((value) => !value)}
          >
            {voiceEnabled ? <Volume2 /> : <VolumeX />}
          </button>
          <button
            type="button"
            className={splitView ? 'is-active' : ''}
            aria-label={splitView ? 'Usar ventana flotante' : 'Dividir pantalla'}
            aria-pressed={splitView}
            onClick={() => {
              setSplitView((value) => !value);
              setReferenceFocus(false);
            }}
          >
            {splitView ? <Minimize2 /> : <Maximize2 />}
          </button>
          <button
            type="button"
            className={`primary-control ${recording || practicing ? 'is-stop' : ''}`}
            aria-label={recording ? 'Detener grabación' : practicing ? 'Detener práctica' : 'Iniciar práctica'}
            disabled={(referenceBusy && !recording) || practiceStarting}
            onClick={() => recording ? stopReferenceRecording() : void startPractice()}
          >
            {recording ? <Square className="fill-current" /> : practicing ? <Pause className="fill-current" /> : <Play className="fill-current" />}
          </button>
          <button
            type="button"
            className={referenceSheetOpen ? 'is-active' : ''}
            aria-label="Cambiar demostración"
            onClick={() => setReferenceSheetOpen(true)}
          >
            <Video />
          </button>
          <button
            type="button"
            className={settingsSheetOpen ? 'is-active' : ''}
            aria-label="Abrir opciones"
            onClick={() => setSettingsSheetOpen(true)}
          >
            <CircleHelp />
          </button>
        </nav>
      </section>

      <Input
        ref={fileInputRef}
        type="file"
        accept="video/*,.mp4,.webm,.mov,.m4v"
        className="hidden"
        aria-label="Elegir video de demostración"
        disabled={referenceBusy || practicing || practiceStarting}
        onChange={(event) => {
          const file = event.currentTarget.files?.[0];
          event.currentTarget.value = '';
          if (file) void uploadReference(file);
        }}
      />

      <Sheet open={referenceSheetOpen} onOpenChange={setReferenceSheetOpen}>
        <SheetContent side="bottom" className="coach-sheet" showCloseButton={!uploading}>
          <SheetHeader>
            <SheetTitle>Demostración</SheetTitle>
            <SheetDescription>{referenceProfile ? 'Referencia lista' : 'Grabá o subí el movimiento correcto.'}</SheetDescription>
          </SheetHeader>
          {uploading ? (
            <div className="sheet-progress" aria-live="polite">
              <LoaderCircle className="animate-spin" />
              <p>Analizando video</p>
              <Progress value={uploadProgress} aria-label="Progreso del análisis" />
              <strong>{uploadProgress}%</strong>
              <Button variant="outline" disabled={uploadProgress === 100} onClick={() => uploadAbortRef.current?.abort()}>
                {uploadProgress === 100 ? 'Guardando…' : 'Cancelar'}
              </Button>
            </div>
          ) : (
            <div className="sheet-actions">
              <Button
                onClick={() => fileInputRef.current?.click()}
                disabled={referenceBusy || practicing || practiceStarting}
                className="sheet-primary-action"
              >
                <Upload /> Subir video
              </Button>
              <Button
                onClick={() => void startReferenceRecording()}
                disabled={(referenceBusy && !recording) || practicing || practiceStarting}
                variant="outline"
              >
                <Radio /> {referenceProfile ? 'Grabar otra' : 'Grabar'}
              </Button>
            </div>
          )}
        </SheetContent>
      </Sheet>

      <Sheet open={settingsSheetOpen} onOpenChange={setSettingsSheetOpen}>
        <SheetContent side="bottom" className="coach-sheet">
          <SheetHeader>
            <SheetTitle>Opciones</SheetTitle>
            <SheetDescription>Ajustes de la sesión.</SheetDescription>
          </SheetHeader>
          <div className="settings-list">
            <label className="settings-row">
              <span>Nivel</span>
              <select value={level} onChange={(event) => setLevel(event.target.value as TrainingLevel)} aria-label="Nivel de exigencia">
                <option value="beginner">Principiante</option>
                <option value="pro">Avanzado</option>
              </select>
            </label>
            <label className="settings-row">
              <span>Feedback por voz</span>
              <Switch checked={voiceEnabled} disabled={!voiceSupported} onCheckedChange={setVoiceEnabled} />
            </label>
            <label className="settings-row">
              <span>Vista dividida</span>
              <Switch checked={splitView} onCheckedChange={(checked) => { setSplitView(checked); setReferenceFocus(false); }} />
            </label>
            <label className="settings-row">
              <span>Puntos del instructor</span>
              <Switch checked={showInstructorPose} onCheckedChange={setShowInstructorPose} />
            </label>
          </div>
          <div className="settings-actions">
            <Button variant="outline" onClick={resetSession}><RotateCcw /> Reiniciar sesión</Button>
            <Button variant="outline" onClick={installApp}><Download /> Instalar app</Button>
          </div>
          <p className="privacy-note"><ShieldCheck /> El análisis sucede en este dispositivo.</p>
        </SheetContent>
      </Sheet>
    </main>
  );
}

function ModelLoading() {
  return (
    <div className="absolute inset-0 z-20 grid place-items-center bg-[#080d12]/85 backdrop-blur-sm">
      <div className="text-center">
        <LoaderCircle className="mx-auto size-7 animate-spin text-lime-300" />
        <p className="mt-3 text-sm font-medium text-white">Cargando análisis corporal</p>
        <p className="mt-1 text-xs text-slate-500">Solo sucede la primera vez.</p>
      </div>
    </div>
  );
}

function preferredMimeType() {
  const candidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate)) ?? '';
}

function cloneLandmarks(landmarks: NormalizedLandmark[]) {
  return landmarks.map(({ x, y, z, visibility }) => ({ x, y, z, visibility }));
}

function syncCanvas(canvas: HTMLCanvasElement, video: HTMLVideoElement) {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth) canvas.width = video.videoWidth;
  if (canvas.height !== video.videoHeight) canvas.height = video.videoHeight;
}

function clearCanvas(canvas: HTMLCanvasElement) {
  canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
}

function cameraErrorMessage(error: unknown) {
  if (error instanceof DOMException && error.name === 'NotAllowedError') {
    return 'Necesito permiso para usar la cámara. Habilitalo en la configuración del navegador.';
  }
  if (error instanceof DOMException && error.name === 'NotFoundError') {
    return 'No encontré una cámara disponible en este dispositivo.';
  }
  return 'No pude iniciar la cámara o el análisis. Revisá la conexión e intentá nuevamente.';
}
