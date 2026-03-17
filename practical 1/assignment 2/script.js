async function train(lr) {
    const xs = tf.tensor([1,2,3,4]);
    const ys = tf.tensor([2,4,6,8]);

    const model = tf.sequential();
    model.add(tf.layers.dense({units:1, inputShape:[1]}));

    model.compile({
        optimizer: tf.train.sgd(lr),
        loss: 'meanSquaredError'
    });

    const history = await model.fit(xs, ys, {epochs:50});
    return history.history.loss[49];
}

async function run() {
    let a = await train(0.01);
    let b = await train(0.1);
    let c = await train(0.5);

    document.getElementById("result").innerText =
        `0.01 → ${a}
         0.1 → ${b}
         0.5 → ${c}`;
}