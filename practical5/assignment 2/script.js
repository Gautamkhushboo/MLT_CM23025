let model;
let isModelLoaded = false;

const webcam = document.getElementById("webcam");
const overlay = document.getElementById("overlay");

// Load model
async function loadModel() {
  overlay.innerText = "⏳ Loading Model...";
  
  model = await mobilenet.load();

  isModelLoaded = true;
  overlay.innerText = "✅ Model Loaded";
  console.log("Model Loaded");
}

loadModel();

// Start Camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
    overlay.innerText = "📷 Camera Started";
  } catch (err) {
    alert("❌ Camera permission denied!");
  }
}

// Start Detection
function startDetection() {
  if (!isModelLoaded) {
    alert("⏳ Please wait, model is still loading...");
    return;
  }

  detect();
}

// Detection loop
function detect() {
  setInterval(async () => {
    const predictions = await model.classify(webcam);

    const label = predictions[0].className;
    const prob = (predictions[0].probability * 100).toFixed(2);

    overlay.innerText = `${label} (${prob}%)`;
  }, 500);
}