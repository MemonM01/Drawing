import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("video");
const drawCanvas = document.getElementById("draw");
const hudCanvas = document.getElementById("hud");
const drawCtx = drawCanvas.getContext("2d");
const hudCtx = hudCanvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");

let lastPoint = null;
const ERASER_RADIUS = 30;

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

// Pinch: thumb tip close to index tip
function isPinching(lm) {
  return dist(lm[4], lm[8]) < 0.055; // slightly easier
}

// More forgiving "open palm":
// fingertips should be above their PIP joints (for selfie camera view this still works well)
function isOpenPalmSimple(lm) {
  const indexUp = lm[8].y < lm[6].y;
  const middleUp = lm[12].y < lm[10].y;
  const ringUp = lm[16].y < lm[14].y;
  const pinkyUp = lm[20].y < lm[18].y;

  // also require fingers somewhat spread (optional but helps)
  const spread = dist(lm[8], lm[20]) > 0.25;

  return indexUp && middleUp && ringUp && pinkyUp && spread;
}

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

function hudText(text) {
  hudCtx.save();
  hudCtx.font = "20px system-ui, Arial";
  hudCtx.fillStyle = "rgba(255,255,255,0.95)";
  hudCtx.textAlign = "left";
  hudCtx.fillText(text, 16, 34);
  hudCtx.restore();
}

function drawHudCursor(p, mode) {
  hudCtx.save();
  hudCtx.lineWidth = 3;
  hudCtx.beginPath();
  hudCtx.arc(p.x, p.y, mode === "erase" ? ERASER_RADIUS : 10, 0, Math.PI * 2);
  hudCtx.strokeStyle =
    mode === "draw" ? "rgba(255,255,255,0.95)" :
    mode === "erase" ? "rgba(255,255,255,0.95)" :
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
        const palm = isOpenPalmSimple(lm);

        if (palm && !pinch) {
          hudText("PALM = ERASE");
          eraseAt(indexTipPx);
          drawHudCursor(indexTipPx, "erase");
          lastPoint = null;
        } else if (pinch) {
          hudText("PINCH = DRAW");
          drawHudCursor(indexTipPx, "draw");
          if (lastPoint) drawLine(lastPoint, indexTipPx);
          lastPoint = indexTipPx;
        } else {
          hudText("IDLE");
          drawHudCursor(indexTipPx, "idle");
          lastPoint = null;
        }
      } else {
        hudText("NO HAND");
        lastPoint = null;
      }
    }

    requestAnimationFrame(loop);
  };

  requestAnimationFrame(loop);
}

clearBtn?.addEventListener("click", () => {
  drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
});

run().catch((e) => {
  console.error(e);
  alert("Failed to start. Check console for the error.");
});