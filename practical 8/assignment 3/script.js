let model1, model2;
let loaded = false;

// Load two MobileNet versions
async function loadModels() {

  document.getElementById("status").innerText = "⏳ Loading Model 1...";
  model1 = await mobilenet.load({version: 1, alpha: 1.0});

  document.getElementById("status").innerText = "⏳ Loading Model 2...";
  model2 = await mobilenet.load({version: 2, alpha: 1.0});

  loaded = true;
  document.getElementById("status").innerText = "✅ Models Ready!";
}

loadModels();

// Image preview
imageUpload.addEventListener("change", function() {
  const file = this.files[0];
  if (file) {
    preview.src = URL.createObjectURL(file);
  }
});

// Compare
compareBtn.addEventListener("click", async () => {

  if (!loaded) {
    alert("Wait for models to load!");
    return;
  }

  if (!preview.src) {
    alert("Upload image first!");
    return;
  }

  const p1 = await model1.classify(preview);
  const p2 = await model2.classify(preview);

  document.getElementById("model1").innerText =
    "MobileNet V1: " + p1[0].className + " (" + (p1[0].probability*100).toFixed(2) + "%)";

  document.getElementById("model2").innerText =
    "MobileNet V2: " + p2[0].className + " (" + (p2[0].probability*100).toFixed(2) + "%)";
});