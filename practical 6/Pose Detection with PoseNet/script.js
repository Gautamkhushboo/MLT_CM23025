alert("JS Loaded");
let video = document.getElementById("video");
let canvas = document.getElementById("canvas");
let ctx = canvas.getContext("2d");

let net;
let running = false;

// ✅ Load model FIRST
async function loadModel() {
  net = await posenet.load();
  console.log("✅ Model Loaded");
}

// ✅ Start camera AFTER model loads
async function startCamera() {
  if (!net) {
    alert("Model not loaded yet!");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });

  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();

    // ✅ FIX: wait before using video size
    setTimeout(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (canvas.width === 0 || canvas.height === 0) {
        alert("Camera not ready. Click Start again.");
        return;
      }

      running = true;
      detectPose();
    }, 500);
  };
}async function startCamera() {
  if (!net) {
    alert("Model not loaded yet!");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });

  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();

    // ✅ FIX: wait before using video size
    setTimeout(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (canvas.width === 0 || canvas.height === 0) {
        alert("Camera not ready. Click Start again.");
        return;
      }

      running = true;
      detectPose();
    }, 500);
  };
}async function startCamera() {
  if (!net) {
    alert("Model not loaded yet!");
    return;
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: true
  });

  video.srcObject = stream;

  video.onloadedmetadata = () => {
    video.play();

    // ✅ FIX: wait before using video size
    setTimeout(() => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      if (canvas.width === 0 || canvas.height === 0) {
        alert("Camera not ready. Click Start again.");
        return;
      }

      running = true;
      detectPose();
    }, 500);
  };
}

function stopCamera() {
  running = false;
  let stream = video.srcObject;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
  }
}

// ✅ SAFE pose detection loop
async function detectPose() {

  // ✅ FIX (new line)
  if (video.videoWidth === 0 || video.videoHeight === 0) {
    requestAnimationFrame(detectPose);
    return;
  }

  if (!running || !net) return;

  const mode = document.getElementById("mode").value;
  const confidence = parseFloat(document.getElementById("confidence").value);

  let poses = [];

  try {
    if (mode === "single") {
      const pose = await net.estimateSinglePose(video);
      poses = [pose];
    } else {
      poses = await net.estimateMultiplePoses(video);
    }
  } catch (err) {
    console.error("Pose error:", err);
    return;
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  poses.forEach(pose => {
    drawKeypoints(pose.keypoints, confidence);
    drawSkeleton(pose.keypoints, confidence);
  });

  document.getElementById("poseCount").innerText =
    "Poses Detected: " + poses.length;

  requestAnimationFrame(detectPose);
}
 
function drawKeypoints(keypoints, threshold) {
  keypoints.forEach(point => {
    if (point.score > threshold) {
      ctx.beginPath();
      ctx.arc(point.position.x, point.position.y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  });
}

function drawSkeleton(keypoints, threshold) {
  const adjacent = posenet.getAdjacentKeyPoints(keypoints, threshold);

  adjacent.forEach(pair => {
    ctx.beginPath();
    ctx.moveTo(pair[0].position.x, pair[0].position.y);
    ctx.lineTo(pair[1].position.x, pair[1].position.y);
    ctx.strokeStyle = "lime";
    ctx.lineWidth = 2;
    ctx.stroke();
  });
}

// ✅ Load model automatically on page load
loadModel();