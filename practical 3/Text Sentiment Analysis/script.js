function analyze() {

  const text = document.getElementById("textInput").value.toLowerCase();

  const positive = ["good", "great", "amazing", "love", "excellent"];
  const negative = ["bad", "worst", "hate", "poor", "terrible"];

  let score = 0;

  positive.forEach(word => {
    if (text.includes(word)) score++;
  });

  negative.forEach(word => {
    if (text.includes(word)) score--;
  });

  let result = "";

  if (score > 0) result = "😊 Positive";
  else if (score < 0) result = "😡 Negative";
  else result = "😐 Neutral";

  document.getElementById("result").innerText = result;
}