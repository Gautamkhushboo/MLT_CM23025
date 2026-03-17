function calculate() {
    const v1 = tf.tensor([1, 2, 3]);
    const v2 = tf.tensor([4, 5, 6]);

    const add = v1.add(v2);
    const mul = v1.mul(v2);

    document.getElementById("output").innerText =
        "Vector 1: " + v1.toString() + "\n" +
        "Vector 2: " + v2.toString() + "\n\n" +
        "Addition: " + add.toString() + "\n\n" +
        "Multiplication: " + mul.toString();
}