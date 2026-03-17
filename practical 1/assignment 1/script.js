async function runModel() {
    const xs = tf.tensor([1,2,3,4,5]);
    const ys = tf.tensor([2,4,6,8,10]);

    const model = tf.sequential();
    model.add(tf.layers.dense({units:1, inputShape:[1]}));

    model.compile({
        optimizer: tf.train.sgd(0.1),
        loss: 'meanSquaredError'
    });

    await model.fit(xs, ys, {epochs:100});

    const preds = model.predict(xs).dataSync();

    new Chart(document.getElementById("chart"), {
        type: 'line',
        data: {
            labels: [1,2,3,4,5],
            datasets: [
                {label: "Actual", data: ys.dataSync()},
                {label: "Predicted", data: preds}
            ]
        }
    });
}