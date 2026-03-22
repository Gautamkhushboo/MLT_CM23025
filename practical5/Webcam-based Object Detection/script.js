let model;
let webcam = document.getElementById("webcam");
let statusText = document.getElementById("status");
let resultText = document.getElementById("result");

// Load MobileNet model
async function loadModel() {
    model = await mobilenet.load();
    statusText.innerHTML = "✅ Model Loaded";
}

loadModel();

// Start Camera
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: true
        });
        webcam.srcObject = stream;
    } catch (error) {
        alert("❌ Camera permission denied!");
        console.error(error);
    }
}

// Start Detection
function startDetection() {

    if (!model) {
        alert("Model not loaded yet!");
        return;
    }

    statusText.innerHTML = "🔍 Detecting...";

    setInterval(async () => {

        const predictions = await model.classify(webcam);

        resultText.innerHTML =
            "🧠 " + predictions[0].className +
            " (" + (predictions[0].probability * 100).toFixed(2) + "%)";

    }, 1000);
}