let model;

// LOAD MODEL
async function loadModel() {
    try {
        model = await tf.loadLayersModel('localstorage://my-model');
        document.getElementById("status").innerText = "✅ Model Loaded!";
    } catch {
        alert("❌ No saved model found! Run Assignment 1 first.");
    }
}

// PREDICT
async function predict() {

    if (!model) {
        alert("Load model first!");
        return;
    }

    const val = document.getElementById("inputValue").value;

    if (val === "") {
        alert("Enter a number!");
        return;
    }

    const input = tf.tensor2d([Number(val)], [1,1]);
    const output = model.predict(input);

    const result = await output.data();

    document.getElementById("result").innerText =
        "Result: " + result[0].toFixed(2);
}