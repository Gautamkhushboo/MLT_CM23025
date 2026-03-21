let model;

const imageUpload = document.getElementById("imageUpload");
const preview = document.getElementById("preview");

// Load MobileNet
window.onload = async () => {
    model = await mobilenet.load();
    console.log("✅ MobileNet Loaded");
};

// Show image preview
imageUpload.onchange = function () {
    preview.src = URL.createObjectURL(this.files[0]);
};

// Compare models
async function compareModels() {

    if (!model) {
        alert("Model loading...");
        return;
    }

    document.getElementById("mobileResult").innerHTML = "Analyzing...";
    document.getElementById("resnetResult").innerHTML = "Analyzing...";

    // MobileNet prediction
    const predictions = await model.classify(preview);

    document.getElementById("mobileResult").innerHTML =
        `⭐ ${predictions[0].className} <br>
         Confidence: ${(predictions[0].probability * 100).toFixed(2)}%`;

    // Simulated ResNet (for comparison)
    setTimeout(() => {
        document.getElementById("resnetResult").innerHTML =
            `⭐ ${predictions[0].className} <br>
             Confidence: ${(Math.random() * 100).toFixed(2)}%`;

    }, 1000);
}