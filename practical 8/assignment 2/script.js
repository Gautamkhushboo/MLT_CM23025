let model;
let correct = 0;
let total = 0;

async function loadModel() {
  model = await mobilenet.load();
  console.log("✅ Model Loaded");
}

document.addEventListener("DOMContentLoaded", () => {

  document.getElementById("testBtn").addEventListener("click", async () => {

    const fileInput = document.getElementById("imageUpload");
    const actualInput = document.getElementById("actual");

    if (!fileInput.files[0]) {
      alert("Upload image first!");
      return;
    }

    if (!actualInput.value) {
      alert("Enter actual label!");
      return;
    }

    const img = new Image();
    img.src = URL.createObjectURL(fileInput.files[0]);

    img.onload = async () => {

      const pred = await model.classify(img);
      const predicted = pred[0].className.toLowerCase();
      const actual = actualInput.value.toLowerCase();

      total++;

      if (predicted.includes(actual)) correct++;

      document.getElementById("accuracy").innerText =
        "Accuracy: " + ((correct / total) * 100).toFixed(2) + "%";
    };

  });

});

loadModel();