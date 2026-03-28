let model;

// TRAIN MODEL (Assignment 1)
async function trainModel() {

    document.getElementById("status").innerText = "Training model...";

    model = tf.sequential();

    model.add(tf.layers.dense({
        units: 1,
        inputShape: [1]
    }));

    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    });

    // Simple dataset: y = 2x
    const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
    const ys = tf.tensor2d([2, 4, 6, 8], [4, 1]);

    await model.fit(xs, ys, { epochs: 200 });

    document.getElementById("status").innerText = "✅ Model Trained!";
}

// SAVE MODEL (Assignment 1)
async function saveModel() {
    if (!model) {
        alert("Train model first!");
        return;
    }

    await model.save('localstorage://my-model');

    document.getElementById("status").innerText = "💾 Model Saved!";
}

// LOAD MODEL (Assignment 2)
async function loadModel() {
    try {
        model = await tf.loadLayersModel('localstorage://my-model');
        document.getElementById("status").innerText = "📂 Model Loaded!";
    } catch {
        alert("No saved model found!");
    }
}

// PREDICT
async function predict() {

    if (!model) {
        alert("⚠️ Please click 'Train Model' first!");
        return;
    }

    const val = document.getElementById("inputValue").value;

    if (val === "") {
        alert("Enter a number!");
        return;
    }

    const input = tf.tensor2d([Number(val)], [1, 1]);
    const output = model.predict(input);
    const result = await output.data();

    document.getElementById("result").innerText =
        "Result: " + result[0].toFixed(2);
}async function predict() {

    if (!model) {
        alert("⚠️ Please click 'Train Model' first!");
        return;
    }

    const val = document.getElementById("inputValue").value;

    if (val === "") {
        alert("Enter a number!");
        return;
    }

    const input = tf.tensor2d([Number(val)], [1, 1]);
    const output = model.predict(input);
    const result = await output.data();

    document.getElementById("result").innerText =
        "Result: " + result[0].toFixed(2);
}async function predict() {

    if (!model) {
        alert("⚠️ Please click 'Train Model' first!");
        return;
    }

    const val = document.getElementById("inputValue").value;

    if (val === "") {
        alert("Enter a number!");
        return;
    }

    const input = tf.tensor2d([Number(val)], [1, 1]);
    const output = model.predict(input);
    const result = await output.data();

    document.getElementById("result").innerText =
        "Result: " + result[0].toFixed(2);
}