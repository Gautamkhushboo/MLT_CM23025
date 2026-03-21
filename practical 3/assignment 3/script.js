let denseModel, rnnModel;

// Dataset
const sentences = [
    "I love this", "Amazing product", "Very happy",
    "I hate this", "Very bad", "Worst experience"
];

const labels = [1,1,1,0,0,0];

// Convert text → numbers
function encode(text){
    return text.split('').map(c => c.charCodeAt(0) / 255);
}

// Padding
function pad(seq, maxLen = 20){
    while(seq.length < maxLen) seq.push(0);
    return seq.slice(0, maxLen);
}

// TRAIN MODELS
async function trainModels(){

    document.getElementById("status").innerText = "⏳ Training...";

    const xs = tf.tensor2d(sentences.map(s => pad(encode(s))));
    const ys = tf.tensor2d(labels, [labels.length,1]);

    // 🔹 Dense Model
    denseModel = tf.sequential();
    denseModel.add(tf.layers.dense({inputShape:[20], units:16, activation:'relu'}));
    denseModel.add(tf.layers.dense({units:1, activation:'sigmoid'}));

    denseModel.compile({
        optimizer:'adam',
        loss:'binaryCrossentropy'
    });

    await denseModel.fit(xs, ys, {epochs:30});

    // 🔹 RNN Model
    rnnModel = tf.sequential();
    rnnModel.add(tf.layers.simpleRNN({
        units:16,
        inputShape:[20,1]
    }));
    rnnModel.add(tf.layers.dense({units:1, activation:'sigmoid'}));

    rnnModel.compile({
        optimizer:'adam',
        loss:'binaryCrossentropy'
    });

    const xsRNN = xs.reshape([xs.shape[0],20,1]);

    await rnnModel.fit(xsRNN, ys, {epochs:30});

    document.getElementById("status").innerText = "✅ Models trained!";
}

// COMPARE
async function compareModels(){

    const text = document.getElementById("inputText").value;

    if(!denseModel || !rnnModel){
        alert("Train models first!");
        return;
    }

    const input = tf.tensor2d([pad(encode(text))]);
    const inputRNN = input.reshape([1,20,1]);

    const dPred = (await denseModel.predict(input).data())[0];
    const rPred = (await rnnModel.predict(inputRNN).data())[0];

    document.getElementById("denseResult").innerText =
        "Dense: " + (dPred > 0.5 ? "😊 Positive" : "😡 Negative");

    document.getElementById("rnnResult").innerText =
        "RNN: " + (rPred > 0.5 ? "😊 Positive" : "😡 Negative");
}