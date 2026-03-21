let model;

// Dataset
const data = [
    { text: "I love this product", label: 1 },
    { text: "This is amazing", label: 1 },
    { text: "Very happy with service", label: 1 },
    { text: "Absolutely fantastic experience", label: 1 },
    { text: "I hate this", label: 0 },
    { text: "Worst product ever", label: 0 },
    { text: "Very bad service", label: 0 },
    { text: "Terrible experience", label: 0 }
];

// Convert text → number (simple feature)
function feature(text) {
    return text.length;
}

async function trainModel() {

    const status = document.getElementById("trainStatus");
    const bar = document.getElementById("progressBar");

    status.innerText = "⏳ Training started...";
    bar.style.width = "0%";

    // Prepare data
    const xs = tf.tensor(data.map(d => [feature(d.text)]));
    const ys = tf.tensor(data.map(d => [d.label]));

    // Model
    model = tf.sequential();
    model.add(tf.layers.dense({
        units: 8,
        inputShape: [1],
        activation: 'relu'
    }));
    model.add(tf.layers.dense({
        units: 1,
        activation: 'sigmoid'
    }));

    model.compile({
        optimizer: 'adam',
        loss: 'binaryCrossentropy',
        metrics: ['accuracy']
    });

    // Train
    await model.fit(xs, ys, {
        epochs: 50,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                let progress = ((epoch + 1) / 50) * 100;
                bar.style.width = progress + "%";

                status.innerText =
                    `Epoch ${epoch + 1}/50 | Loss: ${logs.loss.toFixed(4)}`;
            }
        }
    });

    status.innerText = "✅ Model trained successfully!";
}