let model;

// Load MobileNet model
async function loadModel() {
    model = await mobilenet.load();
    console.log("✅ MobileNet Loaded");
}

loadModel();

// Show image preview
document.getElementById("imageUpload").addEventListener("change", function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function() {
        const img = document.getElementById("preview");
        img.src = reader.result;
        img.style.display = "block";
    };

    if (file) {
        reader.readAsDataURL(file);
    }
});

// Classify Image
async function classifyImage() {

    const resultDiv = document.getElementById("result");
    const img = document.getElementById("preview");

    if (!model) {
        resultDiv.innerHTML = "<p>⚠️ Model loading... please wait</p>";
        return;
    }

    if (!img.src) {
        resultDiv.innerHTML = "<p>⚠️ Please upload an image</p>";
        return;
    }

    resultDiv.innerHTML = "<p>⏳ Analyzing...</p>";

    const predictions = await model.classify(img);

    resultDiv.innerHTML = "";

    predictions.slice(0, 3).forEach((pred, index) => {
        resultDiv.innerHTML += `
            <p> ${index + 1}. ${pred.className} 
            (${(pred.probability * 100).toFixed(2)}%)</p>
        `;
    });
}