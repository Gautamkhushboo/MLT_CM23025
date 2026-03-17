function run() {
    const tensor = tf.tensor([[1, 2], [3, 4]]);

    const reshaped = tensor.reshape([4, 1]);
    const flattened = tensor.flatten();

    document.getElementById("output").innerText =
        "Original Tensor:\n" + tensor.toString() + "\n\n" +
        "Reshaped (4x1):\n" + reshaped.toString() + "\n\n" +
        "Flattened (1D):\n" + flattened.toString() + "\n\n" +
        "Difference:\nreshape changes shape, flatten converts to 1D";
}