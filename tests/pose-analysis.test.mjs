import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  assessPose, calculatePoseMetrics, drawPose, INITIAL_REP_STATE,
  selectPoseSide, sideConnections, SIDE_POINTS,
} from '../lib/pose-analysis.ts';

const good = { trunk: 'good', thigh: 'good', shin: 'good' };
function landmarks() {
  const points = Array.from({ length: 33 }, (_, i) => ({ x: 0.1 + i * 0.02, y: 0.5, z: 0, visibility: 0.99 }));
  points[0].y = 0.1;
  points[11].y = 0.2;
  points[12].y = 0.25;
  points[25].y = 0.65;
  points[26].y = 0.6;
  points[27].y = 0.8;
  points[28].y = 0.75;
  points[31].y = 0.9;
  points[32].y = 0.8;
  return points;
}

function drawingCanvas() {
  const dots = [];
  const lines = [];
  let line;
  let dot;
  const context = {
    clearRect() {}, save() {}, restore() {},
    beginPath() { line = null; dot = null; },
    moveTo(x, y) { line = { from: [x, y] }; },
    lineTo(x, y) { line.to = [x, y]; },
    arc(x, y) { dot = { at: [x, y] }; },
    fill() { dots.push({ ...dot, color: this.fillStyle }); },
    stroke() { if (line) lines.push({ ...line, color: this.strokeStyle }); },
  };
  return { canvas: { width: 1000, height: 1000, getContext: () => context }, dots, lines };
}

test('notebook lateral selection uses shoulder-to-foot height, not average visibility', () => {
  const points = landmarks();
  Object.values(SIDE_POINTS.left).forEach((index) => { points[index].visibility = 0.6; });
  assert.equal(selectPoseSide(points), 'left');
  assert.equal(calculatePoseMetrics(points).side, 'left');
  points[32].y = 1;
  assert.equal(selectPoseSide(points), 'right');
  assert.equal(calculatePoseMetrics(points).side, 'right');
  points[12].y = points[11].y;
  points[32].y = points[31].y;
  assert.equal(selectPoseSide(points), 'right');
  assert.equal(selectPoseSide([]), null);
});

test('each lateral consists of exactly seven notebook indices and six unilateral segments', () => {
  assert.deepEqual(Object.values(SIDE_POINTS.left), [11, 13, 15, 23, 25, 27, 31]);
  assert.deepEqual(Object.values(SIDE_POINTS.right), [12, 14, 16, 24, 26, 28, 32]);
  for (const side of ['left', 'right']) {
    const indices = Object.values(SIDE_POINTS[side]);
    assert.equal(sideConnections(side).length, 6);
    for (const edge of sideConnections(side)) {
      assert.ok(indices.includes(edge.start));
      assert.ok(indices.includes(edge.end));
    }
  }
});

test('renders only the selected seven points, all green for a correct pose', () => {
  const points = landmarks();
  for (const side of ['left', 'right']) {
    const { canvas, dots, lines } = drawingCanvas();
    drawPose(canvas, points, good, false, side);
    assert.equal(dots.length, 7);
    assert.equal(lines.length, 6);
    assert.deepEqual(dots.map((dot) => dot.at), Object.values(SIDE_POINTS[side]).map((index) => [points[index].x * 1000, points[index].y * 1000]));
    assert.ok([...dots, ...lines].every(({ color }) => color === '#bef264'));
  }
});

test('only the affected leg segments and shared joints become red', () => {
  const points = landmarks();
  const { canvas, dots, lines } = drawingCanvas();
  drawPose(canvas, points, { ...good, shin: 'bad' });
  assert.equal(lines.filter(({ color }) => color === '#fb7185').length, 2);
  assert.deepEqual(dots.filter(({ color }) => color === '#fb7185').map((dot) => dot.at), [25, 27, 31].map((index) => [points[index].x * 1000, points[index].y * 1000]));
  assert.equal(dots.filter(({ color }) => color === '#bef264').length, 4);
});

test('reference playback uses the same unilateral seven-point drawing', () => {
  const { canvas, dots, lines } = drawingCanvas();
  drawPose(canvas, landmarks(), { trunk: 'bad', thigh: 'bad', shin: 'bad' }, true);
  assert.equal(dots.length, 7);
  assert.equal(lines.length, 6);
  assert.ok([...dots, ...lines].every(({ color }) => color === '#bef264'));
});

test('pixel-correct angles are invariant between portrait and landscape frames', () => {
  const points = landmarks();
  const original = calculatePoseMetrics(points, 1);
  const wide = calculatePoseMetrics(points.map((point) => ({ ...point, x: point.x / 2 })), 2);
  const tall = calculatePoseMetrics(points.map((point) => ({ ...point, y: point.y / 2 })), 0.5);
  for (const key of ['offsetAngle', 'trunkAngle', 'thighAngle', 'shinAngle']) {
    assert.ok(Math.abs(original[key] - wide[key]) < 1e-8);
    assert.ok(Math.abs(original[key] - tall[key]) < 1e-8);
  }
});

test('a missing or out-of-frame foot never produces a green posture assessment', () => {
  for (const mutation of [{ visibility: 0.1 }, { x: 1.1 }]) {
    const points = landmarks();
    Object.assign(points[31], mutation);
    const assessment = assessPose(calculatePoseMetrics(points), null, 'beginner', INITIAL_REP_STATE);
    assert.equal(assessment.aligned, false);
    assert.equal(assessment.state, null);
    assert.equal(assessment.primary.title, 'Ajustá el encuadre');
    assert.ok(Object.values(assessment.segments).every((state) => state === 'warning'));
  }
});
