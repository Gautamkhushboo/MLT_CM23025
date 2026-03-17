function createTensors() {
    const scalar = tf.scalar(5);
    const vector = tf.tensor([1, 2, 3]);
    const matrix = tf.tensor([[1, 2], [3, 4]]);

    document.getElementById("output").innerText =
        "Scalar: " + scalar.toString() + "\n\n" +
        "Vector: " + vector.toString() + "\n\n" +
        "Matrix: " + matrix.toString();
}