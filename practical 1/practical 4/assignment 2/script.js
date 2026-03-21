let model;

// Load model
window.onload = async () => {
    model = await mobilenet.load();
    console.log("✅ Model Loaded");
};

async function classifyAll() {

    const images = document.querySelectorAll(".testImg");
    const results = document.querySelectorAll(".result");

    if (!model) {
        alert("Model still loading...");
        return;
    }

    for (let i = 0; i < images.length; i++) {

        results[i].innerHTML = "🔍 Analyzing...";

        const predictions = await model.classify(images[i]);

        results[i].innerHTML = `
            ⭐ ${predictions[0].className} <br>
            (${(predictions[0].probability * 100).toFixed(2)}%)
        `;
    }
}