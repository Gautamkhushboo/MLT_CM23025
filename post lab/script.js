const video = document.getElementById("video");
const canvas = document.getElementById("overlay");
const emotionText = document.getElementById("emotion");
const confidenceText = document.getElementById("confidence");
const chartCanvas = document.getElementById("emotionChart");

let chart;
let emotionHistory = [];

// 🎭 Emoji map
const emojiMap = {
    happy: "😄",
    sad: "😢",
    angry: "😠",
    surprised: "😲",
    neutral: "😐",
    fearful: "😨",
    disgusted: "🤢"
};

// 🧠 Meaning
const emotionMeaning = {
    happy: "You are in a joyful mood!",
    sad: "You might be facing some stress.",
    angry: "Try to relax and stay calm.",
    surprised: "Something unexpected happened!",
    neutral: "You look calm and normal.",
    fearful: "Stay calm, everything is okay.",
    disgusted: "You seem uncomfortable."
};

// 🎤 Voice messages
const voiceMessages = {
    happy: "You look happy! Keep smiling!",
    sad: "You seem sad. Take a deep breath and relax.",
    angry: "You look angry. Try to calm down.",
    surprised: "You look surprised!",
    neutral: "You look calm.",
    fearful: "You look scared. Stay relaxed.",
    disgusted: "You seem uncomfortable."
};

let lastSpokenEmotion = "";
let lastSpeakTime = 0;

// 🎤 Speak function
function speak(text) {
    const now = Date.now();

    // avoid repeating too often
    if (now - lastSpeakTime < 4000) return;

    const speech = new SpeechSynthesisUtterance(text);
    speech.lang = "en-US";
    speech.rate = 1;

    window.speechSynthesis.speak(speech);
    lastSpeakTime = now;
}

// Load models
async function loadModels() {
    const URL = "https://justadudewhohacks.github.io/face-api.js/models";

    await faceapi.nets.tinyFaceDetector.loadFromUri(URL);
    await faceapi.nets.faceExpressionNet.loadFromUri(URL);
}

// Start camera
async function startVideo() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
}

// Chart
function initChart() {
    chart = new Chart(chartCanvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [{
                label: "Emotion Confidence",
                data: [],
                borderWidth: 3,
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            plugins: {
                legend: {
                    labels: { color: "black" }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: "Time (Emotion)" }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: "Confidence" }
                }
            }
        }
    });
}

// Detection
function startDetection() {

    video.addEventListener("loadedmetadata", () => {

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        setInterval(async () => {

            const detections = await faceapi
                .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
                .withFaceExpressions();

            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (detections.length > 0) {

                const det = detections[0];
                const expressions = det.expressions;

                let bestEmotion = "neutral";
                let maxVal = 0;

                for (const exp in expressions) {
                    if (expressions[exp] > maxVal) {
                        maxVal = expressions[exp];
                        bestEmotion = exp;
                    }
                }

                const box = det.detection.box;

                // 🎭 Emoji on face
                ctx.font = "50px Arial";
                ctx.fillText(
                    emojiMap[bestEmotion],
                    box.x + box.width / 3,
                    box.y - 10
                );

                // 📊 Graph update
                emotionHistory.push({
                    value: maxVal,
                    label: bestEmotion
                });

                if (emotionHistory.length > 15) emotionHistory.shift();

                chart.data.labels = emotionHistory.map(
                    (e, i) => i + " (" + e.label + ")"
                );

                chart.data.datasets[0].data =
                    emotionHistory.map(e => e.value);

                chart.update();

                // 🧠 Text output
                emotionText.innerText =
                    emojiMap[bestEmotion] + " " + emotionMeaning[bestEmotion];

                confidenceText.innerText =
                    "Confidence: " + (maxVal * 100).toFixed(2) + "%";

                // 🎤 Voice output (only when emotion changes)
                if (bestEmotion !== lastSpokenEmotion) {
                    speak(voiceMessages[bestEmotion]);
                    lastSpokenEmotion = bestEmotion;
                }

            } else {
                emotionText.innerText = "No face detected";
                confidenceText.innerText = "";
            }

        }, 400);
    });
}

// INIT
async function init() {
    await loadModels();
    await startVideo();
    initChart();
    startDetection();
}

init();