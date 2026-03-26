let model;
const webcam = document.getElementById("webcam");
const resultText = document.getElementById("result");

// Load MobileNet
async function loadModel() {
  model = await mobilenet.load();
  console.log("✅ Model Loaded");
}

loadModel();

// Start Camera
async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcam.srcObject = stream;
  } catch (err) {
    alert("❌ Camera permission denied!");
  }
}

// Start Prediction
function startPrediction() {
  if (!model) {
    alert("Model not loaded yet!");
    return;
  }

  setInterval(async () => {
    const predictions = await model.classify(webcam);

    const label = predictions[0].className;
    const prob = (predictions[0].probability * 100).toFixed(2);

    resultText.innerText = `Result: ${label} (${prob}%)`;
  }, 1000);
}