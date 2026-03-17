let model;

async function train() {
    const xs = tf.tensor([1,2,3,4,5]);
    const ys = tf.tensor([2,4,6,8,10]);

    model = tf.sequential();
    model.add(tf.layers.dense({units:1, inputShape:[1]}));

    model.compile({
        optimizer: tf.train.sgd(0.1),
        loss: 'meanSquaredError'
    });

    await model.fit(xs, ys, {epochs:100});
}

async function predict() {
    if (!model) await train();

    let val = document.getElementById("inputVal").value;

    const result = model.predict(tf.tensor([Number(val)]));

    document.getElementById("output").innerText =
        "Prediction: " + result.dataSync();
}