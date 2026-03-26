let model;
let webcam;
let lastTime = 0;

async function loadModel() {
    model = await mobilenet.load();
    console.log("Model Loaded");
}

loadModel();

// Start camera
async function startCamera() {
    webcam = document.getElementById("webcam");

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        webcam.srcObject = stream;
    } catch (error) {
        alert("❌ Camera permission denied!");
    }
}

// Start detection + FPS calculation
function startDetection() {
    detectFrame();
}

async function detectFrame() {
    if (!model || !webcam) {
        alert("Model or camera not ready!");
        return;
    }

    const now = performance.now();
    const fps = Math.round(1000 / (now - lastTime));
    lastTime = now;

    const predictions = await model.classify(webcam);

    document.getElementById("label").innerText =
        `📌 ${predictions[0].className} (${(predictions[0].probability * 100).toFixed(2)}%)`;

    document.getElementById("fps").innerText = `⚡ FPS: ${fps}`;

    requestAnimationFrame(detectFrame);
}