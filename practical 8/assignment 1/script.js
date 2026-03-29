let model;

async function loadModel() {
  model = await mobilenet.load();
}

document.getElementById("predictBtn").onclick = async () => {
  const file = imageUpload.files[0];
  const img = document.getElementById("preview");

  img.src = URL.createObjectURL(file);

  img.onload = async () => {
    const predictions = await model.classify(img);

    result.innerHTML = "";
    predictions.slice(0,3).forEach(p => {
      let li = document.createElement("li");
      li.innerText = `${p.className} (${(p.probability*100).toFixed(2)}%)`;
      result.appendChild(li);
    });
  };
};

loadModel();