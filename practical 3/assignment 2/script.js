let model;

// Dataset
const data = [
    { text: "I love this product", label: 1 },
    { text: "Amazing experience", label: 1 },
    { text: "Very happy", label: 1 },
    { text: "Worst product", label: 0 },
    { text: "I hate this", label: 0 },
    { text: "Very bad experience", label: 0 }
];

// Feature (simple demo)
function feature(text) {
    return text.length;
}

// TRAIN MODEL
async function trainModel() {

    const status = document.getElementById("trainStatus");
    status.innerText = "⏳ Training...";

    const xs = tf.tensor(data.map(d => [feature(d.text)]));
    const ys = tf.tensor(data.map(d => [d.label]));

    model = tf.sequential();
    model.add(tf.layers.dense({ units: 8, inputShape: [1], activation: 'relu' }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    await model.fit(xs, ys, { epochs: 40 });

    status.innerText = "✅ Model trained successfully!";
}

// PREDICT
async function predictSentiment() {

    const text = document.getElementById("inputText").value;
    const resultText = document.getElementById("resultText");
    const bar = document.getElementById("confidenceBar");

    if (!model) {
        resultText.innerText = "⚠️ Train model first!";
        return;
    }

    if (text.trim() === "") {
        resultText.innerText = "⚠️ Enter some text!";
        return;
    }

    const input = tf.tensor([[text.length]]);
    const prediction = model.predict(input);
    const value = (await prediction.data())[0];

    let sentiment, confidence, color;

    if (value > 0.5) {
        sentiment = "😊 Positive";
        confidence = value * 100;
        color = "#00ffcc";
    } else {
        sentiment = "😡 Negative";
        confidence = (1 - value) * 100;
        color = "#ff4d4d";
    }

    resultText.innerText = `${sentiment} (${confidence.toFixed(2)}%)`;

    bar.style.width = confidence + "%";
    bar.style.background = color;
}