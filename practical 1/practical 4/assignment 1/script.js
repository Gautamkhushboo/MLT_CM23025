let model;

// Load model
window.onload = async () => {
    model = await mobilenet.load();
    console.log("✅ Model Loaded");
};

async function classify() {

    const img = document.getElementById("testImage");
    const output = document.getElementById("output");

    if (!model) {
        output.innerHTML = "⏳ Loading model...";
        return;
    }

    if (!img.complete) {
        output.innerHTML = "⚠️ Image not loaded";
        return;
    }

    output.innerHTML = "🔍 Analyzing...";

    try {
        const predictions = await model.classify(img);

        output.innerHTML = "";

        predictions.slice(0,3).forEach((p, i) => {
            output.innerHTML += `
                <p>⭐ ${p.className} 
                (${(p.probability * 100).toFixed(2)}%)</p>
            `;
        });

    } catch (error) {
        console.error(error);
        output.innerHTML = "❌ Error during prediction";
    }
}