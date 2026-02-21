import * as vision from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const { FilesetResolver, HandLandmarker } = vision;

const video = document.getElementById("video");
const drawCanvas = document.getElementById("draw");
const hudCanvas = document.getElementById("hud");
const drawCtx = drawCanvas.getContext("2d");
const hudCtx = hudCanvas.getContext("2d");

const clearBtn = document.getElementById("clearBtn");

function resizeCanvasesToVideo() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  drawCanvas.width = w;
  drawCanvas.height = h;
  hudCanvas.width = w;
  hudCanvas.height = h;
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function toPixel(pt, canvas) {
  return { x: pt.x * canvas.width, y: pt.y * canvas.height };
}

// --- Gesture heuristics ---
// Pinch: thumb tip (4) close to index tip (8)
function isPinching(lm) {
  const pinch = dist(lm[4], lm[8]);          // normalized
  return pinch < 0.05;                       // tweak: 0.04–0.06
}

// Finger extended test (robust-ish): compare distance to wrist
// If tip is farther from wrist than PIP, finger is likely extended.
function fingerExtended(lm, tipIdx, pipIdx) {
  const wrist = lm[0];
  return dist(lm[tipIdx], wrist) > dist(lm[pipIdx], wrist);
}

// Fist: all 4 fingers (index/middle/ring/pinky) NOT extended,
// and thumb NOT extended as well (roughly).
function isFist(lm) {
  const indexExt = fingerExtended(lm, 8, 6);
  const middleExt = fingerExtended(lm, 12, 10);
  const ringExt = fingerExtended(lm, 16, 14);
  const pinkyExt = fingerExtended(lm, 20, 18);

  // Thumb: tip (4) vs IP (3)
  const thumbExt = fingerExtended(lm, 4, 3);

  return !indexExt && !middleExt && !ringExt && !pinkyExt && !thumbExt;
}

// Draw settings
let lastPoint = null;

// Eraser settings
const ERASER_RADIUS = 28; // pixels (increase/decrease)

function drawLine(from, to) {
  drawCtx.lineWidth = 6;
  drawCtx.lineCap = "round";
  drawCtx.strokeStyle = "white";

  drawCtx.beginPath();
  drawCtx.moveTo(from.x, from.y);
  drawCtx.lineTo(to.x, to.y);
  drawCtx.stroke();
}

function eraseAt(p) {
  // Smooth eraser using compositing
  drawCtx.save();
  drawCtx.globalCompositeOperation = "destination-out";
  drawCtx.beginPath();
  drawCtx.arc(p.x, p.y, ERASER_RADIUS, 0, Math.PI * 2);
  drawCtx.fill();
  drawCtx.restore();
}

function hudClear() {
  hudCtx.clearRect(0, 0, hudCanvas.width, hudCanvas.height);
}

function drawHudCursor(p, mode) {
  // mode: "draw" | "erase" | "idle"
  hudCtx.save();
  hudCtx.lineWidth = 3;
  hudCtx.beginPath();
  hudCtx.arc(p.x, p.y, mode === "erase" ? ERASER_RADIUS : 10, 0, Math.PI * 2);
  hudCtx.strokeStyle =
    mode === "draw" ? "rgba(255,255,255,0.9)" :
    mode === "erase" ? "rgba(255,255,255,0.9)" :
    "rgba(255,255,255,0.35)";
  hudCtx.stroke();
  hudCtx.restore();
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user" },
    audio: false
  });
  video.srcObject = stream;

  await new Promise((res) => (video.onloadedmetadata = res));
  resizeCanvasesToVideo();
}

async function createHandLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  return await HandLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });
}

async function run() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Camera API not supported in this browser.");
    return;
  }

  await setupCamera();
  const landmarker = await createHandLandmarker();

  let lastVideoTime = -1;

  const loop = () => {
    const now = performance.now();

    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hudClear();

      const result = landmarker.detectForVideo(video, now);
      const hands = result.landmarks;

      if (hands && hands.length > 0) {
        const lm = hands[0];

        const indexTipPx = toPixel(lm[8], drawCanvas);

        const fist = isFist(lm);
        const pinch = isPinching(lm);

        if (fist) {
          // ERASE
          eraseAt(indexTipPx);
          drawHudCursor(indexTipPx, "erase");
          lastPoint = null;
        } else if (pinch) {
          // DRAW
          drawHudCursor(indexTipPx, "draw");
          if (lastPoint) drawLine(lastPoint, indexTipPx);
          lastPoint = indexTipPx;
        } else {
          // idle (tracking but not drawing)
          drawHudCursor(indexTipPx, "idle");
          lastPoint = null;
        }
      } else {
        lastPoint = null;
      }
    }

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

clearBtn.addEventListener("click", () => {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});

run().catch((e) => {
  console.error(e);
  alert("Failed to start. Open DevTools console for details.");
});
