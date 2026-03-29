let model;

async function loadModel() {
    model = await mobilenet.load();
    console.log("✅ MobileNet Loaded");
}

document.getElementById("predictBtn").addEventListener("click", async () => {

    const file = document.getElementById("imageUpload").files[0];

    if (!file) {
        alert("⚠️ Please upload an image first!");
        return;
    }

    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(file);

    img.onload = async () => {

        const predictions = await model.classify(img);
        const top = predictions[0];

        document.getElementById("result").innerText =
            "Prediction: " + top.className;

        document.getElementById("barFill").style.width =
            (top.probability * 100).toFixed(2) + "%";
    };

});

loadModel();