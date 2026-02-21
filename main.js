import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

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

// Pinch: thumb tip (4) close to index tip (8)
function isPinching(lm) {
  const pinch = dist(lm[4], lm[8]); // normalized
  return pinch < 0.05;              // tweak: 0.04–0.06
}

// Finger extended heuristic: compare distance to wrist.
// If tip is farther from wrist than PIP, finger is likely extended.
function fingerExtended(lm, tipIdx, pipIdx) {
  const wrist = lm[0];
  return dist(lm[tipIdx], wrist) > dist(lm[pipIdx], wrist);
}

// Open palm: all fingers extended
function isOpenPalm(lm) {
  const index = fingerStraight(lm, 5, 6, 7, 8);
  const middle = fingerStraight(lm, 9, 10, 11, 12);
  const ring = fingerStraight(lm, 13, 14, 15, 16);
  const pinky = fingerStraight(lm, 17, 18, 19, 20);

  // Optional: require hand to be "spread" (helps avoid false positives)
  const spread = dist(lm[8], lm[20]) > 0.35; // index tip to pinky tip

  return index && middle && ring && pinky && spread;
}

// Draw settings
let lastPoint = null;

// Eraser settings
const ERASER_RADIUS = 28;

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
  await video.play().catch(() => {}); // helps iPhone/Safari sometimes
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

        const openPalm = isOpenPalm(lm);
        const pinch = isPinching(lm);

        if (openPalm) {
          eraseAt(indexTipPx);
          drawHudCursor(indexTipPx, "erase");
          lastPoint = null;
        } else if (pinch) {
          drawHudCursor(indexTipPx, "draw");
          if (lastPoint) drawLine(lastPoint, indexTipPx);
          lastPoint = indexTipPx;
        } else {
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