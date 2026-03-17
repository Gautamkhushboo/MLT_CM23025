"use strict";

Chart.defaults.color = "#8a8fa8";
Chart.defaults.borderColor = "#252830";
Chart.defaults.font.family = "'JetBrains Mono', monospace";
Chart.defaults.font.size = 11;

const ACCENT  = "#00e5ff";
const ORANGE  = "#f97316";
const GREEN   = "#10b981";
const RED     = "#f43f5e";
const YELLOW  = "#fbbf24";
const COLORS  = [ACCENT, ORANGE, RED, GREEN, YELLOW, "#a78bfa", "#fb7185"];

function chartOptions() {
  return {
    responsive: true,
    maintainAspectRatio: true,
    plugins: { legend: { labels: { color: "#8a8fa8", boxWidth: 12 } } },
    scales: {
      x: { grid: { color: "#1a1d24" }, ticks: { color: "#4a4f65", maxTicksLimit: 8 } },
      y: { grid: { color: "#1a1d24" }, ticks: { color: "#4a4f65", maxTicksLimit: 6 } }
    }
  };
}

function destroyChart(c) { if (c && c.destroy) c.destroy(); }

// Tab switching
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

// Generate synthetic data: y = m*x + b + noise
function generateData(n = 100, noise = 0.2, m = 2, b = 1, xMin = 0, xMax = 10) {
  const xs = [], ys = [];
  for (let i = 0; i < n; i++) {
    const x = xMin + Math.random() * (xMax - xMin);
    const y = m * x + b + (Math.random() - 0.5) * 2 * noise * (xMax - xMin);
    xs.push(x); ys.push(y);
  }
  return { xs, ys };
}

function shuffleTogether(arr1, arr2) {
  const idx = arr1.map((_, i) => i).sort(() => Math.random() - 0.5);
  return { a: idx.map(i => arr1[i]), b: idx.map(i => arr2[i]) };
}

function trainTestSplit(xs, ys, trainRatio = 0.8) {
  const { a: sx, b: sy } = shuffleTogether(xs, ys);
  const cut = Math.floor(sx.length * trainRatio);
  return { trainX: sx.slice(0, cut), trainY: sy.slice(0, cut), testX: sx.slice(cut), testY: sy.slice(cut) };
}

function buildModel(lr = 0.1) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
  model.compile({ optimizer: tf.train.sgd(lr), loss: "meanSquaredError" });
  return model;
}

function normalise(arr) {
  const min = Math.min(...arr), max = Math.max(...arr);
  return { norm: arr.map(v => (v - min) / (max - min)), min, max };
}

function denormalise(normArr, min, max) {
  return normArr.map(v => v * (max - min) + min);
}

function r2Score(actual, predicted) {
  const mean = actual.reduce((s, v) => s + v, 0) / actual.length;
  const ss_tot = actual.reduce((s, v) => s + (v - mean) ** 2, 0);
  const ss_res = actual.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  return 1 - ss_res / ss_tot;
}

/* ── MAIN PRACTICAL ── */
let mainDataChart = null, mainLossChart = null;

document.getElementById("main-train-btn").addEventListener("click", async () => {
  const n = +document.getElementById("main-samples").value;
  const epochs = +document.getElementById("main-epochs").value;
  const lr = +document.getElementById("main-lr").value;
  const noise = +document.getElementById("main-noise").value;
  const btn = document.getElementById("main-train-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="training-pulse"></span>Training...';

  const { xs, ys } = generateData(n, noise, 2, 1);
  const { norm: normX, min: xMin, max: xMax } = normalise(xs);
  const { norm: normY, min: yMin, max: yMax } = normalise(ys);
  const xT = tf.tensor2d(normX, [normX.length, 1]);
  const yT = tf.tensor2d(normY, [normY.length, 1]);
  const model = buildModel(lr);
  const lossHistory = [];
  const logBox = document.getElementById("main-log");
  logBox.innerHTML = "";

  await model.fit(xT, yT, {
    epochs, batchSize: 32,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        lossHistory.push(logs.loss);
        if (epoch % Math.max(1, Math.floor(epochs / 20)) === 0 || epoch === epochs - 1) {
          const line = document.createElement("div");
          line.className = "log-line";
          line.innerHTML = `<span class="epoch">Epoch ${String(epoch+1).padStart(4," ")}/${epochs}</span>  Loss: <span class="loss-val">${logs.loss.toFixed(6)}</span>`;
          logBox.appendChild(line);
          logBox.scrollTop = logBox.scrollHeight;
        }
      }
    }
  });

  const lineX = Array.from({ length: 100 }, (_, i) => i / 99);
  const linePredNorm = Array.from(model.predict(tf.tensor2d(lineX, [100, 1])).dataSync());
  const lineXD = denormalise(lineX, xMin, xMax);
  const lineYD = denormalise(linePredNorm, yMin, yMax);

  destroyChart(mainDataChart);
  mainDataChart = new Chart(document.getElementById("main-data-chart").getContext("2d"), {
    data: {
      datasets: [
        { type: "scatter", label: "Training Data", data: xs.map((x, i) => ({ x, y: ys[i] })), backgroundColor: "rgba(0,229,255,0.25)", pointRadius: 3, borderColor: "transparent" },
        { type: "line", label: "Regression Line", data: lineXD.map((x, i) => ({ x, y: lineYD[i] })), borderColor: ORANGE, borderWidth: 2.5, pointRadius: 0 }
      ]
    },
    options: { ...chartOptions(), scales: { x: { type: "linear", grid: { color: "#1a1d24" }, ticks: { color: "#4a4f65" } }, y: { grid: { color: "#1a1d24" }, ticks: { color: "#4a4f65" } } } }
  });

  destroyChart(mainLossChart);
  mainLossChart = new Chart(document.getElementById("main-loss-chart").getContext("2d"), {
    type: "line",
    data: { labels: lossHistory.map((_, i) => i + 1), datasets: [{ label: "MSE Loss", data: lossHistory, borderColor: ACCENT, backgroundColor: "rgba(0,229,255,0.06)", borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3 }] },
    options: chartOptions()
  });

  const weights = model.getWeights();
  const w = weights[0].dataSync()[0], b = weights[1].dataSync()[0];
  const realSlope = w * (yMax - yMin) / (xMax - xMin)