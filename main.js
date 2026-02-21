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

// ---------- Gestures ----------

// Pinch: thumb tip (4) close to index tip (8)
function isPinching(lm) {
  return dist(lm[4], lm[8]) < 0.05; // tweak 0.04–0.06
}

// Angle at point b between ba and bc
function angleDeg(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };
  const dot = ab.x * cb.x + ab.y * cb.y;
  const magAB = Math.hypot(ab.x, ab.y);
  const magCB = Math.hypot(cb.x, cb.y);
  const cos = dot / (magAB * magCB + 1e-9);
  const clamped = Math.max(-1, Math.min(1, cos));
  return (Math.acos(clamped) * 180) / Math.PI;
}

// Finger straight if joints are near-straight
function fingerStraight(lm, mcp, pip, dip, tip) {
  const a1 = angleDeg(lm[mcp], lm[pip], lm[dip]);
  const a2 = angleDeg(lm[pip], lm[dip], lm[tip]);
  return a1 > 160 && a2 > 160; // tweak 150–170 if needed
}

// Open palm: four fingers straight (+ optional spread)
function isOpenPalm(lm) {
  const index = fingerStraight(lm, 5, 6, 7, 8);
  const middle = fingerStraight(lm, 9, 10, 11, 12);
  const ring = fingerStraight(lm, 13, 14, 15, 16);
  const pinky = fingerStraight(lm, 17, 18, 19, 20);

  // Spread check helps avoid false positives (optional)
  const spread = dist(lm[8], lm[20]) > 0.30; // easier than 0.35

  return index && middle && ring && pinky && spread;
}

// ---------- Drawing / Erasing ----------

let lastPoint = null;
const ERASER_RADIUS = 30;

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
  await video.play().catch(() => {});
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

        const pinch = isPinching(lm);
        const openPalm = isOpenPalm(lm);

        // Erase wins over draw
        if (openPalm && !pinch) {
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