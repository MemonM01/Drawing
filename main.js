import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";

const video = document.getElementById("video");
const drawCanvas = document.getElementById("draw");
const hudCanvas = document.getElementById("hud");
const drawCtx = drawCanvas.getContext("2d");
const hudCtx = hudCanvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");

let lastPoint = null;
const ERASER_RADIUS = 32;

// --- Utils ---
function resizeCanvasesToVideo() {
  const w = video.videoWidth;
  const h = video.videoHeight;
  if (!w || !h) return;
  if (drawCanvas.width !== w || drawCanvas.height !== h) {
    drawCanvas.width = w;
    drawCanvas.height = h;
    hudCanvas.width = w;
    hudCanvas.height = h;
  }
}

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

function toPixel(pt, canvas) {
  return { x: pt.x * canvas.width, y: pt.y * canvas.height };
}

// --- Gestures (more forgiving thresholds) ---
function pinchDistance(lm) {
  return dist(lm[4], lm[8]); // thumb tip to index tip (normalized)
}

function isPinching(lm) {
  // 0.05 can be too strict on some cameras. Start at 0.085.
  return pinchDistance(lm) < 0.085;
}

// Simple fist: fingertips below PIP joints (y bigger) for all 4 fingers
function isFistSimple(lm) {
  const indexDown = lm[8].y > lm[6].y;
  const middleDown = lm[12].y > lm[10].y;
  const ringDown = lm[16].y > lm[14].y;
  const pinkyDown = lm[20].y > lm[18].y;
  return indexDown && middleDown && ringDown && pinkyDown;
}

// --- Drawing / erasing ---
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

function hudText(lines) {
  hudCtx.save();
  hudCtx.font = "18px system-ui, Arial";
  hudCtx.fillStyle = "rgba(255,255,255,0.95)";
  hudCtx.textAlign = "left";
  let y = 28;
  for (const line of lines) {
    hudCtx.fillText(line, 14, y);
    y += 22;
  }
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
    numHands: 1,
    // These help detection in tricky lighting / webcams
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
}

async function run() {
  await setupCamera();
  const landmarker = await createHandLandmarker();

  let lastVideoTime = -1;

  const loop = () => {
    const now = performance.now();

    // Ensure canvases match actual video size
    resizeCanvasesToVideo();

    if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      hudClear();

      const result = landmarker.detectForVideo(video, now);
      const hands = result?.landmarks;

      const handCount = hands?.length ?? 0;

      if (handCount > 0) {
        const lm = hands[0];
        const p = toPixel(lm[8], drawCanvas);

        const pd = pinchDistance(lm);
        const pinch = isPinching(lm);
        const fist = isFistSimple(lm);

        let mode = "IDLE";
        if (fist) mode = "FIST=ERASE";
        else if (pinch) mode = "PINCH=DRAW";

        hudText([
          `HANDS: ${handCount}`,
          `pinchDist: ${pd.toFixed(3)} (thresh 0.085)`,
          `mode: ${mode}`
        ]);

        if (fist) {
          eraseAt(p);
          drawHudCursor(p, "erase");
          lastPoint = null;
        } else if (pinch) {
          drawHudCursor(p, "draw");
          if (lastPoint) drawLine(lastPoint, p);
          lastPoint = p;
        } else {
          drawHudCursor(p, "idle");
          lastPoint = null;
        }
      } else {
        hudText([`HANDS: 0`, `No landmarks detected`]);
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