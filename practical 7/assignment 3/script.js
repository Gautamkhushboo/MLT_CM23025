let model;

// TRAIN MODEL
async function trainModel() {
    model = tf.sequential();

    model.add(tf.layers.dense({units: 1, inputShape: [1]}));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    const xs = tf.tensor([1,2,3,4,5]);
    const ys = tf.tensor([2,4,6,8,10]);

    document.getElementById("status").innerText = "Training...";

    await model.fit(xs, ys, {epochs: 150});

    document.getElementById("status").innerText = "✅ Model Trained!";
}

// SAVE MODEL AS FILES
async function saveModel() {
    if (!model) {
        alert("Train model first!");
        return;
    }

    await model.save('downloads://my-model');

    document.getElementById("status").innerText = "📥 Model Downloaded!";
}

// LOAD MODEL FROM FILES
async function loadModel() {
    const files = document.getElementById("upload").files;

    if (files.length < 2) {
        alert("Upload both model.json and weights.bin");
        return;
    }

    model = await tf.loadLayersModel(
        tf.io.browserFiles([files[0], files[1]])
    );

    document.getElementById("status").innerText = "✅ Model Loaded!";
}

// PREDICT
async function predict() {
    if (!model) {
        alert("Load or train model first!");
        return;
    }

    const val = document.getElementById("input").value;

    const output = model.predict(tf.tensor([Number(val)]));

    const result = await output.data();

    document.getElementById("result").innerText =
        "Result: " + result[0].toFixed(2);
}