import type { NormalizedLandmark } from '@mediapipe/tasks-vision';

export type PoseSide = 'left' | 'right';
export type TrainingLevel = 'beginner' | 'pro';
export type SegmentName = 'trunk' | 'thigh' | 'shin';
export type SegmentState = 'good' | 'bad' | 'warning' | 'neutral';

export interface PoseMetrics {
  side: PoseSide;
  visibility: number;
  offsetAngle: number;
  trunkAngle: number;
  thighAngle: number;
  shinAngle: number;
}

export interface ReferenceFrame {
  t: number;
  metrics: PoseMetrics;
  landmarks: NormalizedLandmark[];
}

export interface ReferenceProfile {
  frames: ReferenceFrame[];
  duration: number;
  maxThighAngle: number;
  createdAt: number;
}

export interface SegmentStates {
  trunk: SegmentState;
  thigh: SegmentState;
  shin: SegmentState;
}

export interface FeedbackItem {
  region: 'camera' | SegmentName | 'general';
  title: string;
  detail: string;
  severity: 'success' | 'warning' | 'error';
}

export interface PoseAssessment {
  segments: SegmentStates;
  primary: FeedbackItem;
  items: Array<{ label: string; value: string; state: SegmentState }>;
  improper: boolean;
  aligned: boolean;
  state: SquatState;
}

export type SquatState = 's1' | 's2' | 's3' | null;

export interface RepState {
  sequence: SquatState[];
  previous: SquatState;
  improper: boolean;
  correct: number;
  incorrect: number;
}

// The seven side-view landmarks used in notebook m06_02 (not the full pose mesh).
export const SIDE_POINTS = {
  left: { shoulder: 11, elbow: 13, wrist: 15, hip: 23, knee: 25, ankle: 27, foot: 31 },
  right: { shoulder: 12, elbow: 14, wrist: 16, hip: 24, knee: 26, ankle: 28, foot: 32 },
} as const;

export function selectPoseSide(landmarks: NormalizedLandmark[]): PoseSide | null {
  const leftShoulder = landmarks[SIDE_POINTS.left.shoulder];
  const leftFoot = landmarks[SIDE_POINTS.left.foot];
  const rightShoulder = landmarks[SIDE_POINTS.right.shoulder];
  const rightFoot = landmarks[SIDE_POINTS.right.foot];
  if (![leftShoulder, leftFoot, rightShoulder, rightFoot].every((point) => point && Number.isFinite(point.y))) return null;
  // Notebook criterion: compare the vertical shoulder-to-foot distance.
  const leftHeight = Math.abs(leftFoot.y - leftShoulder.y);
  const rightHeight = Math.abs(rightFoot.y - rightShoulder.y);
  return leftHeight > rightHeight ? 'left' : 'right';
}

export function sideConnections(side: PoseSide) {
  const points = SIDE_POINTS[side];
  return [
    { start: points.shoulder, end: points.elbow, region: null },
    { start: points.elbow, end: points.wrist, region: null },
    { start: points.shoulder, end: points.hip, region: 'trunk' },
    { start: points.hip, end: points.knee, region: 'thigh' },
    { start: points.knee, end: points.ankle, region: 'shin' },
    { start: points.ankle, end: points.foot, region: 'shin' },
  ] as const;
}

const REGION_LABELS: Record<SegmentName, string> = {
  trunk: 'Espalda',
  thigh: 'Cadera',
  shin: 'Rodilla',
};

export const INITIAL_REP_STATE: RepState = {
  sequence: [],
  previous: null,
  improper: false,
  correct: 0,
  incorrect: 0,
};

function degrees(radians: number) {
  return (radians * 180) / Math.PI;
}

function angleAt(reference: NormalizedLandmark, first: NormalizedLandmark, second: Pick<NormalizedLandmark, 'x' | 'y'>) {
  const a = { x: first.x - reference.x, y: first.y - reference.y };
  const b = { x: second.x - reference.x, y: second.y - reference.y };
  const denominator = Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y);
  if (denominator < Number.EPSILON) return 0;
  const cosine = Math.max(-1, Math.min(1, (a.x * b.x + a.y * b.y) / denominator));
  return degrees(Math.acos(cosine));
}

function sideVisibility(landmarks: NormalizedLandmark[], side: PoseSide) {
  const points = SIDE_POINTS[side];
  // Arms are drawn, but only these five points determine squat tracking quality.
  const indices = [points.shoulder, points.hip, points.knee, points.ankle, points.foot];
  return Math.min(...indices.map((index) => {
    const point = landmarks[index];
    if (!point || point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1) return 0;
    return point.visibility ?? 0;
  }));
}

export function calculatePoseMetrics(landmarks: NormalizedLandmark[], frameAspectRatio = 1): PoseMetrics | null {
  if (landmarks.length < 33) return null;
  const side = selectPoseSide(landmarks);
  if (!side) return null;
  const points = SIDE_POINTS[side];
  // The notebook computes angles in pixels. Correct normalized X for the frame
  // ratio so a portrait upload and a landscape webcam use the same geometry.
  const aspectRatio = Number.isFinite(frameAspectRatio) && frameAspectRatio > 0 ? frameAspectRatio : 1;
  const project = (index: number) => ({ ...landmarks[index], x: landmarks[index].x * aspectRatio });
  const shoulder = project(points.shoulder);
  const hip = project(points.hip);
  const knee = project(points.knee);
  const ankle = project(points.ankle);
  const nose = project(0);
  if (!shoulder || !hip || !knee || !ankle || !nose) return null;

  return {
    side,
    visibility: sideVisibility(landmarks, side),
    offsetAngle: angleAt(nose, project(11), project(12)),
    trunkAngle: angleAt(hip, shoulder, { x: hip.x, y: 0 }),
    thighAngle: angleAt(knee, hip, { x: knee.x, y: 0 }),
    shinAngle: angleAt(ankle, knee, { x: ankle.x, y: 0 }),
  };
}

export function buildReferenceProfile(frames: ReferenceFrame[], duration: number): ReferenceProfile {
  const usable = frames.filter((frame) => frame.metrics.visibility >= 0.45 && frame.metrics.offsetAngle <= 38);
  const source = usable.length >= 5 ? usable : frames;
  const bounded = source.length <= 160
    ? source
    : source.filter((_, index) => index % Math.ceil(source.length / 160) === 0);
  const orderedAngles = bounded.map((frame) => frame.metrics.thighAngle).sort((a, b) => a - b);
  const percentileIndex = Math.max(0, Math.floor(orderedAngles.length * 0.94) - 1);
  return {
    frames: bounded,
    duration,
    maxThighAngle: orderedAngles[percentileIndex] ?? 75,
    createdAt: Date.now(),
  };
}

export function getSquatState(thighAngle: number, level: TrainingLevel, previous: SquatState): SquatState {
  if (thighAngle >= 0 && thighAngle <= 32) return 's1';
  if (thighAngle >= 35 && thighAngle <= 65) return 's2';
  const passStart = level === 'pro' ? 80 : 70;
  if (thighAngle >= passStart && thighAngle <= 95) return 's3';
  return previous;
}

export function updateRepState(current: SquatState, assessmentImproper: boolean, state: RepState): RepState {
  const next: RepState = {
    ...state,
    sequence: [...state.sequence],
    improper: state.improper || assessmentImproper,
  };

  if (current === 's2') {
    const s2Count = next.sequence.filter((item) => item === 's2').length;
    if ((!next.sequence.includes('s3') && s2Count === 0) || (next.sequence.includes('s3') && s2Count === 1)) {
      next.sequence.push('s2');
    }
  } else if (current === 's3' && next.sequence.includes('s2') && !next.sequence.includes('s3')) {
    next.sequence.push('s3');
  }

  if (current === 's1' && state.previous !== 's1') {
    const completed = next.sequence.join(',') === 's2,s3,s2';
    if (completed && !next.improper) next.correct += 1;
    else if (next.sequence.includes('s2')) next.incorrect += 1;
    next.sequence = [];
    next.improper = false;
  }

  next.previous = current;
  return next;
}

function closestReference(metrics: PoseMetrics, profile: ReferenceProfile | null) {
  if (!profile?.frames.length) return null;
  return profile.frames.reduce((closest, frame) => {
    const distance = Math.abs(frame.metrics.thighAngle - metrics.thighAngle);
    const closestDistance = Math.abs(closest.metrics.thighAngle - metrics.thighAngle);
    return distance < closestDistance ? frame : closest;
  });
}

export function assessPose(
  metrics: PoseMetrics,
  profile: ReferenceProfile | null,
  level: TrainingLevel,
  rep: RepState,
): PoseAssessment {
  const state = getSquatState(metrics.thighAngle, level, rep.previous);
  if (metrics.visibility < 0.45) {
    return cameraAssessment('Ajustá el encuadre', 'Necesito ver hombro, cadera, rodilla y pie. Orientá la cámara y revisá la luz; alejate solo si todavía quedás fuera.', null);
  }
  if (metrics.offsetAngle > 38) {
    return cameraAssessment('Ponete de perfil', 'Girá el cuerpo hasta que ambos hombros se superpongan en la imagen.', null);
  }

  const reference = closestReference(metrics, profile)?.metrics;
  const trunkTolerance = level === 'pro' ? 7 : 11;
  const shinTolerance = level === 'pro' ? 7 : 12;
  const expectedTrunk = reference?.trunkAngle ?? Math.min(45, Math.max(12, metrics.thighAngle * 0.46));
  const expectedShin = reference?.shinAngle ?? Math.min(level === 'pro' ? 30 : 40, metrics.thighAngle * 0.36);
  const targetDepth = Math.max(level === 'pro' ? 80 : 70, Math.min(92, profile?.maxThighAngle ?? (level === 'pro' ? 82 : 76)));

  const trunkDelta = metrics.trunkAngle - expectedTrunk;
  const shinDelta = metrics.shinAngle - expectedShin;
  const isMovingThroughSquat = state === 's2' || state === 's3';
  const trunkBad = isMovingThroughSquat && Math.abs(trunkDelta) > trunkTolerance;
  const shinBad = isMovingThroughSquat && shinDelta > shinTolerance;
  const tooDeep = metrics.thighAngle > Math.max(96, targetDepth + 10);
  const needsDepth = state === 's2' && rep.sequence.filter((item) => item === 's2').length <= 1 && metrics.thighAngle < targetDepth - 5;
  const thighBad = tooDeep || needsDepth;

  const segments: SegmentStates = {
    trunk: trunkBad ? 'bad' : 'good',
    thigh: thighBad ? 'bad' : 'good',
    shin: shinBad ? 'bad' : 'good',
  };

  let primary: FeedbackItem;
  if (shinBad) {
    primary = {
      region: 'shin',
      title: 'Llevá la rodilla un poco hacia atrás',
      detail: 'Mantenela alineada con el pie y apoyá el peso en todo el pie.',
      severity: 'error',
    };
  } else if (tooDeep) {
    primary = {
      region: 'thigh',
      title: 'No bajes tanto la cadera',
      detail: 'Frená la bajada en la profundidad marcada por la demostración.',
      severity: 'error',
    };
  } else if (needsDepth) {
    primary = {
      region: 'thigh',
      title: 'Bajá la cadera un poco más',
      detail: 'Continuá el descenso con control hasta alcanzar la profundidad de referencia.',
      severity: 'warning',
    };
  } else if (trunkBad) {
    primary = trunkDelta > 0
      ? { region: 'trunk', title: 'Llevá el pecho un poco atrás', detail: 'Acercá el ángulo del torso al de la demostración.', severity: 'warning' }
      : { region: 'trunk', title: 'Incliná apenas el torso', detail: 'Acompañá la bajada manteniendo la espalda neutra.', severity: 'warning' };
  } else {
    primary = {
      region: 'general',
      title: 'Muy bien, mantené esa posición',
      detail: 'Los segmentos analizados están dentro del rango de la demostración.',
      severity: 'success',
    };
  }

  return {
    segments,
    primary,
    aligned: true,
    improper: shinBad || tooDeep,
    state,
    items: (Object.keys(segments) as SegmentName[]).map((region) => ({
      label: REGION_LABELS[region],
      value: segments[region] === 'good' ? 'En rango' : region === primary.region ? 'Corregir ahora' : 'Revisar',
      state: segments[region],
    })),
  };
}

function cameraAssessment(title: string, detail: string, state: SquatState): PoseAssessment {
  return {
    segments: { trunk: 'warning', thigh: 'warning', shin: 'warning' },
    primary: { region: 'camera', title, detail, severity: 'warning' },
    items: [
      { label: 'Encuadre', value: 'Ajustar', state: 'warning' },
      { label: 'Vista lateral', value: 'Ajustar', state: 'warning' },
      { label: 'Detección', value: 'En espera', state: 'neutral' },
    ],
    improper: false,
    aligned: false,
    state,
  };
}

export function drawPose(
  canvas: HTMLCanvasElement,
  landmarks: NormalizedLandmark[],
  segments: SegmentStates,
  reference = false,
  side = selectPoseSide(landmarks),
) {
  const context = canvas.getContext('2d');
  if (!context) return;
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  if (!side) return;
  const connections = sideConnections(side);

  const colorFor = (state: SegmentState) => {
    if (reference) return '#bef264';
    if (state === 'bad') return '#fb7185';
    if (state === 'warning') return '#fbbf24';
    if (state === 'neutral') return '#64748b';
    return '#bef264';
  };
  const fallbackState: SegmentState = Object.values(segments).some((state) => state === 'warning')
    ? 'warning'
    : 'good';

  for (const connection of connections) {
    const start = landmarks[connection.start];
    const end = landmarks[connection.end];
    if (!start || !end || Math.min(start.visibility ?? 1, end.visibility ?? 1) < 0.35) continue;
    const region = connection.region;
    const state = region ? segments[region] : reference ? 'good' : fallbackState;
    const color = colorFor(state);
    context.save();
    context.beginPath();
    context.moveTo(start.x * width, start.y * height);
    context.lineTo(end.x * width, end.y * height);
    context.lineCap = 'round';
    context.lineWidth = region ? 6 : 4;
    context.strokeStyle = color;
    context.globalAlpha = 0.95;
    context.shadowColor = color;
    context.shadowBlur = region ? 12 : 4;
    context.stroke();
    context.restore();
  }

  Object.values(SIDE_POINTS[side]).forEach((index) => {
    const point = landmarks[index];
    if (!point || (point.visibility ?? 1) < 0.35) return;
    const regions = connections
      .filter((connection) => connection.start === index || connection.end === index)
      .map((connection) => connection.region)
      .filter((region): region is SegmentName => region !== null);
    const badRegion = regions.find((region) => segments[region] === 'bad');
    const warningRegion = regions.find((region) => segments[region] === 'warning');
    const state = badRegion ? 'bad' : warningRegion ? 'warning' : regions.length ? 'good' : fallbackState;
    const color = colorFor(state);
    const radius = 5.5;
    context.save();
    context.beginPath();
    context.arc(point.x * width, point.y * height, radius, 0, Math.PI * 2);
    context.fillStyle = color;
    context.globalAlpha = 1;
    context.shadowColor = color;
    context.shadowBlur = regions.length ? 12 : 5;
    context.fill();
    context.lineWidth = 2;
    context.strokeStyle = '#071015';
    context.stroke();
    context.restore();
  });
}

export function formatDuration(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(safe / 60)).padStart(2, '0')}:${String(safe % 60).padStart(2, '0')}`;
}
