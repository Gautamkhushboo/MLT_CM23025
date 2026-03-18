/* ====================================================
   MNIST CNN Practical — script.js
   Full CNN: Load data → Build model → Train → Evaluate → Predict
==================================================== */

'use strict';

/* ── Global State ── */
let model       = null;
let trainImages = null;
let trainLabels = null;
let testImages  = null;
let testLabels  = null;
let isTraining  = false;

const lossHistory  = [];
const accHistory   = [];
const valLossHistory = [];
const valAccHistory  = [];

const IMAGE_SIZE    = 784;   // 28×28
const NUM_CLASSES   = 10;
const IMAGE_W       = 28;
const IMAGE_H       = 28;

/* ─────────────────────────────────────────────────
   SECTION A — DATA LOADING (MNIST via sprite sheet)
───────────────────────────────────────────────── */

const MNIST_IMAGES_URL =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_URL =
  'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS  = 10000;
const NUM_DATASET_ELEMENTS = 65000;

async function loadMNIST() {
  setProgress(5, 'Fetching MNIST image sprite...');
  addLog('Loading MNIST dataset from Google storage...', 'info');

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = async () => {
      setProgress(35, 'Decoding pixel data...');

      const canvas = document.createElement('canvas');
      canvas.width  = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      const imgData = ctx.getImageData(0, 0, img.width, img.height);
      const float32 = new Float32Array(NUM_DATASET_ELEMENTS * IMAGE_SIZE);

      for (let i = 0; i < float32.length; i++) {
        float32[i] = imgData.data[i * 4] / 255.0;
      }

      setProgress(55, 'Fetching labels...');
      try {
        const res    = await fetch(MNIST_LABELS_URL);
        const labels = new Uint8Array(await res.arrayBuffer());

        const numTrain  = parseInt(document.getElementById('cfg-samples').value);
        const numTest   = Math.min(2000, NUM_TEST_ELEMENTS);

        // Train split
        trainImages = tf.tensor4d(
          float32.slice(0, numTrain * IMAGE_SIZE),
          [numTrain, IMAGE_W, IMAGE_H, 1]
        );
        trainLabels = tf.oneHot(
          tf.tensor1d(Array.from(labels.slice(0, numTrain)), 'int32'),
          NUM_CLASSES
        );

        // Test split (taken from end)
        const testOffset = NUM_TRAIN_ELEMENTS;
        testImages = tf.tensor4d(
          float32.slice(testOffset * IMAGE_SIZE, (testOffset + numTest) * IMAGE_SIZE),
          [numTest, IMAGE_W, IMAGE_H, 1]
        );
        testLabels = tf.tensor1d(
          Array.from(labels.slice(testOffset, testOffset + numTest)),
          'int32'
        );

        setProgress(70, 'Data loaded! Building model...');
        addLog(`✔ Loaded ${numTrain} training samples, ${numTest} test samples`, 'success');
        resolve();
      } catch (err) { reject(err); }
    };
    img.onerror = () => reject(new Error('Failed to load MNIST sprite image.'));
    img.src = MNIST_IMAGES_URL;
  });
}

/* ─────────────────────────────────────────────────
   SECTION B — CNN MODEL ARCHITECTURE
───────────────────────────────────────────────── */

function buildModel() {
  const m = tf.sequential();

  // Conv block 1
  m.add(tf.layers.conv2d({
    inputShape: [IMAGE_W, IMAGE_H, 1],
    kernelSize: 3,
    filters: 8,
    activation: 'relu',
    padding: 'same'
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Conv block 2
  m.add(tf.layers.conv2d({
    kernelSize: 3,
    filters: 16,
    activation: 'relu',
    padding: 'same'
  }));
  m.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

  // Classifier
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));

  m.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  addLog('Model architecture:', 'info');
  addLog('  Conv2D(8, 3×3, relu) → MaxPool(2×2)', 'metric');
  addLog('  Conv2D(16, 3×3, relu) → MaxPool(2×2)', 'metric');
  addLog('  Flatten → Dense(10, softmax)', 'metric');
  addLog(`  Total params: ${m.countParams().toLocaleString()}`, 'metric');

  return m;
}

/* ─────────────────────────────────────────────────
   SECTION C — TRAINING
───────────────────────────────────────────────── */

async function startTraining() {
  if (isTraining) return;
  isTraining = true;

  const btnTrain = document.getElementById('btn-train');
  btnTrain.disabled = true;
  btnTrain.textContent = '⏳ Training...';

  document.getElementById('progress-wrap').style.display = 'block';
  document.getElementById('stats-row').style.display     = 'flex';
  document.getElementById('train-log').style.display     = 'block';
  document.getElementById('charts-grid').style.display   = 'grid';

  lossHistory.length = 0;
  accHistory.length  = 0;
  valLossHistory.length = 0;
  valAccHistory.length  = 0;

  addLog('═══ Starting Training ═══', 'epoch');

  try {
    await tf.ready();
    await loadMNIST();

    model = buildModel();
    setProgress(75, 'Training CNN...');

    const epochs    = parseInt(document.getElementById('cfg-epochs').value);
    const batchSize = parseInt(document.getElementById('cfg-batch').value);
    let   epochNum  = 0;

    await model.fit(trainImages, trainLabels, {
      epochs,
      batchSize,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          epochNum = epoch + 1;
          const loss   = logs.loss.toFixed(4);
          const acc    = (logs.acc * 100).toFixed(1);
          const valLoss = logs.val_loss ? logs.val_loss.toFixed(4) : '—';
          const valAcc  = logs.val_acc  ? (logs.val_acc * 100).toFixed(1) + '%' : '—';

          lossHistory.push(parseFloat(loss));
          accHistory.push(parseFloat(acc));
          if (logs.val_loss) valLossHistory.push(parseFloat(valLoss));
          if (logs.val_acc)  valAccHistory.push(parseFloat(logs.val_acc * 100));

          document.getElementById('stat-epoch').textContent   = `${epochNum}/${epochs}`;
          document.getElementById('stat-loss').textContent    = loss;
          document.getElementById('stat-acc').textContent     = acc + '%';
          document.getElementById('stat-valloss').textContent = valLoss;
          document.getElementById('stat-valacc').textContent  = valAcc;

          const pct = 75 + Math.round((epochNum / epochs) * 25);
          setProgress(pct, `Epoch ${epochNum}/${epochs} complete`);

          addLog(`Epoch ${epochNum}/${epochs}`, 'epoch');
          addLog(`  loss: ${loss}  accuracy: ${acc}%`, 'metric');
          if (logs.val_loss) addLog(`  val_loss: ${valLoss}  val_acc: ${valAcc}`, 'metric');

          drawChart('chart-loss', lossHistory, valLossHistory, 'Loss', '#f97316', '#a78bfa');
          drawChart('chart-acc',  accHistory,  valAccHistory,  'Accuracy (%)', '#22d3a5', '#3b82f6');

          await tf.nextFrame();
        },
        onTrainEnd: () => {
          setProgress(100, '✔ Training complete!');
          addLog('═══ Training Complete! ═══', 'success');
          isTraining = false;

          btnTrain.disabled    = false;
          btnTrain.textContent = '⬡ Re-train';

          document.getElementById('btn-eval').disabled    = false;
          document.getElementById('btn-predict').disabled = false;
          document.getElementById('tf-status').textContent = '● Model Ready';
          document.getElementById('tf-status').className  = 'chip chip-green';
        }
      }
    });

  } catch (err) {
    addLog('❌ Error: ' + err.message, 'epoch');
    setProgress(0, 'Error during training.');
    isTraining = false;
    btnTrain.disabled = false;
    btnTrain.innerHTML = '<span>⬡</span> Load Data &amp; Train';
    console.error(err);
  }
}

/* ─────────────────────────────────────────────────
   SECTION D — EVALUATION
───────────────────────────────────────────────── */

async function evaluateModel() {
  if (!model) return;

  document.getElementById('btn-eval').disabled = true;
  document.getElementById('btn-eval').textContent = '⏳ Evaluating...';

  document.getElementById('eval-results').style.display      = 'flex';
  document.getElementById('sample-preds-wrap').style.display = 'block';
  document.getElementById('class-bars-wrap').style.display   = 'block';

  const numTest = testImages.shape[0];

  // Evaluate
  const evalResult = model.evaluate(
    testImages,
    tf.oneHot(testLabels, NUM_CLASSES)
  );
  const testLoss = (await evalResult[0].data())[0];
  const testAcc  = (await evalResult[1].data())[0];

  document.getElementById('eval-acc-num').textContent    = (testAcc * 100).toFixed(1) + '%';
  document.getElementById('eval-loss-num').textContent   = testLoss.toFixed(4);
  document.getElementById('eval-samples-num').textContent = numTest.toLocaleString();

  // Get all predictions for per-class stats
  const predsT     = model.predict(testImages);
  const predsArr   = await predsT.argMax(-1).data();
  const trueArr    = await testLabels.data();

  // Per-class accuracy
  const classCorrect = new Array(10).fill(0);
  const classTotal   = new Array(10).fill(0);
  for (let i = 0; i < trueArr.length; i++) {
    classTotal[trueArr[i]]++;
    if (predsArr[i] === trueArr[i]) classCorrect[trueArr[i]]++;
  }
  renderClassBars(classCorrect, classTotal);

  // Sample predictions — pick 20 random
  const imgData = await testImages.data();
  const indices = [];
  while (indices.length < 20) {
    const r = Math.floor(Math.random() * numTest);
    if (!indices.includes(r)) indices.push(r);
  }
  renderSamplePredictions(imgData, trueArr, predsArr, indices);

  predsT.dispose();
  evalResult.forEach(t => t.dispose());

  document.getElementById('btn-eval').disabled = false;
  document.getElementById('btn-eval').textContent = '◈ Re-evaluate';
}

function renderSamplePredictions(imgData, trueArr, predsArr, indices) {
  const grid = document.getElementById('sample-grid');
  grid.innerHTML = '';

  indices.forEach(idx => {
    const trueLabel = trueArr[idx];
    const predLabel = predsArr[idx];
    const correct   = trueLabel === predLabel;

    // Draw the 28×28 digit onto a canvas
    const c = document.createElement('canvas');
    c.width  = 56;
    c.height = 56;
    c.className = correct ? 'correct' : 'wrong';
    const ctx = c.getContext('2d');
    ctx.imageSmoothingEnabled = false;

    const small = document.createElement('canvas');
    small.width = 28; small.height = 28;
    const sCtx = small.getContext('2d');
    const id = sCtx.createImageData(28, 28);
    for (let p = 0; p < 784; p++) {
      const val = Math.round(imgData[idx * 784 + p] * 255);
      id.data[p * 4]     = val;
      id.data[p * 4 + 1] = val;
      id.data[p * 4 + 2] = val;
      id.data[p * 4 + 3] = 255;
    }
    sCtx.putImageData(id, 0, 0);
    ctx.drawImage(small, 0, 0, 56, 56);

    const labels = document.createElement('div');
    labels.className = 'sample-labels';
    labels.innerHTML =
      `<div class="true-lbl">true: ${trueLabel}</div>` +
      `<div class="${correct ? 'pred-lbl' : 'pred-wrong'}">pred: ${predLabel}</div>`;

    const item = document.createElement('div');
    item.className = 'sample-item';
    item.appendChild(c);
    item.appendChild(labels);
    grid.appendChild(item);
  });
}

function renderClassBars(correct, total) {
  const wrap = document.getElementById('class-bars');
  wrap.innerHTML = '';
  for (let d = 0; d < 10; d++) {
    const acc = total[d] > 0 ? (correct[d] / total[d] * 100) : 0;
    wrap.innerHTML += `
      <div class="class-bar-row">
        <div class="class-bar-lbl">Digit ${d}</div>
        <div class="class-bar-bg">
          <div class="class-bar-fill" style="width:${acc.toFixed(1)}%"></div>
        </div>
        <div class="class-bar-val">${acc.toFixed(1)}%</div>
      </div>`;
  }
}

/* ─────────────────────────────────────────────────
   SECTION E — DRAW & PREDICT CANVAS
───────────────────────────────────────────────── */

const drawCanvas = document.getElementById('draw-canvas');
const drawCtx    = drawCanvas.getContext('2d');
let   drawing    = false;
let   lastX = 0, lastY = 0;

// Setup drawing canvas
drawCtx.fillStyle = '#000';
drawCtx.fillRect(0, 0, 280, 280);
drawCtx.strokeStyle = '#fff';
drawCtx.lineWidth   = 22;
drawCtx.lineCap     = 'round';
drawCtx.lineJoin    = 'round';

function getPos(e, canvas) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  if (e.touches) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top)  * scaleY
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top)  * scaleY
  };
}

drawCanvas.addEventListener('mousedown', e => {
  drawing = true;
  const { x, y } = getPos(e, drawCanvas);
  lastX = x; lastY = y;
  drawCtx.beginPath();
  drawCtx.arc(x, y, 10, 0, Math.PI * 2);
  drawCtx.fillStyle = '#fff';
  drawCtx.fill();
});

drawCanvas.addEventListener('mousemove', e => {
  if (!drawing) return;
  const { x, y } = getPos(e, drawCanvas);
  drawCtx.beginPath();
  drawCtx.moveTo(lastX, lastY);
  drawCtx.lineTo(x, y);
  drawCtx.stroke();
  lastX = x; lastY = y;
  if (model) autoPredict();
});

drawCanvas.addEventListener('mouseup',   () => { drawing = false; });
drawCanvas.addEventListener('mouseleave',() => { drawing = false; });

// Touch support
drawCanvas.addEventListener('touchstart', e => {
  e.preventDefault();
  drawing = true;
  const { x, y } = getPos(e, drawCanvas);
  lastX = x; lastY = y;
}, { passive: false });

drawCanvas.addEventListener('touchmove', e => {
  e.preventDefault();
  if (!drawing) return;
  const { x, y } = getPos(e, drawCanvas);
  drawCtx.beginPath();
  drawCtx.moveTo(lastX, lastY);
  drawCtx.lineTo(x, y);
  drawCtx.stroke();
  lastX = x; lastY = y;
  if (model) autoPredict();
}, { passive: false });

drawCanvas.addEventListener('touchend', () => { drawing = false; });

function clearCanvas() {
  drawCtx.fillStyle = '#000';
  drawCtx.fillRect(0, 0, 280, 280);
  document.getElementById('pred-result').style.display      = 'none';
  document.getElementById('pred-placeholder').style.display = 'flex';
}

/* Auto-predict while drawing (throttled) */
let predictTimer = null;
function autoPredict() {
  clearTimeout(predictTimer);
  predictTimer = setTimeout(predictDrawing, 200);
}

async function predictDrawing() {
  if (!model) return;

  // Show result area
  document.getElementById('pred-placeholder').style.display = 'none';
  document.getElementById('pred-result').style.display      = 'flex';

  const tensor = tf.tidy(() => {
    // Downscale 280×280 → 28×28
    const small = document.createElement('canvas');
    small.width = 28; small.height = 28;
    const sCtx = small.getContext('2d');
    sCtx.imageSmoothingEnabled = true;
    sCtx.drawImage(drawCanvas, 0, 0, 28, 28);

    const imgData = sCtx.getImageData(0, 0, 28, 28);
    const input   = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      // Use red channel (greyscale)
      input[i] = imgData.data[i * 4] / 255.0;
    }
    return tf.tensor4d(input, [1, 28, 28, 1]);
  });

  const probsTensor = model.predict(tensor);
  const probs = Array.from(await probsTensor.data());
  tensor.dispose();
  probsTensor.dispose();

  const predicted = probs.indexOf(Math.max(...probs));
  const confidence = (probs[predicted] * 100).toFixed(1);

  document.getElementById('pred-digit').textContent = predicted;
  document.getElementById('pred-conf').textContent  = confidence + '%';

  // Render probability bars
  const barsWrap = document.getElementById('prob-bars');
  barsWrap.innerHTML = '';
  probs.forEach((p, d) => {
    const pct = (p * 100).toFixed(1);
    const isTop = d === predicted;
    barsWrap.innerHTML += `
      <div class="prob-row">
        <div class="prob-digit-lbl">${d}</div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${isTop ? 'top' : ''}"
               style="width:${pct}%"></div>
        </div>
        <div class="prob-pct">${pct}%</div>
      </div>`;
  });
}

/* ─────────────────────────────────────────────────
   SECTION F — CHART DRAWING (vanilla canvas)
───────────────────────────────────────────────── */

function drawChart(canvasId, data1, data2, label, color1, color2) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const pad = { top: 16, right: 20, bottom: 32, left: 44 };
  const cW = W - pad.left - pad.right;
  const cH = H - pad.top  - pad.bottom;

  ctx.clearRect(0, 0, W, H);

  const allVals = [...data1, ...data2];
  if (!allVals.length) return;

  const minV = Math.min(...allVals) * 0.95;
  const maxV = Math.max(...allVals) * 1.05;
  const range = maxV - minV || 1;

  const toX = i => pad.left + (i / (data1.length - 1 || 1)) * cW;
  const toY = v => pad.top  + (1 - (v - minV) / range) * cH;

  // Grid lines
  ctx.strokeStyle = 'rgba(26,34,53,0.8)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (i / 4) * cH;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + cW, y);
    ctx.stroke();
    const val = maxV - (i / 4) * range;
    ctx.fillStyle = '#3a4e6e';
    ctx.font = '10px Space Mono, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(2), pad.left - 6, y + 4);
  }

  // X labels
  ctx.textAlign = 'center';
  data1.forEach((_, i) => {
    ctx.fillStyle = '#3a4e6e';
    ctx.font = '10px Space Mono, monospace';
    ctx.fillText(i + 1, toX(i), H - 8);
  });
  ctx.fillStyle = '#4a5e7e';
  ctx.fillText('Epoch', pad.left + cW / 2, H - 0);

  // Draw line helper
  function drawLine(data, color) {
    if (data.length < 1) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = 'round';
    data.forEach((v, i) => {
      if (i === 0) ctx.moveTo(toX(i), toY(v));
      else ctx.lineTo(toX(i), toY(v));
    });
    ctx.stroke();

    // Dots
    data.forEach((v, i) => {
      ctx.beginPath();
      ctx.arc(toX(i), toY(v), 4, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    });
  }

  drawLine(data1, color1);
  if (data2.length) drawLine(data2, color2);

  // Legend
  const ly = pad.top + 4;
  ctx.fillStyle = color1;
  ctx.fillRect(W - 100, ly, 12, 3);
  ctx.fillStyle = '#7a90b8';
  ctx.font = '9px Space Mono, monospace';
  ctx.textAlign = 'left';
  ctx.fillText('train', W - 84, ly + 4);

  if (data2.length) {
    ctx.fillStyle = color2;
    ctx.fillRect(W - 100, ly + 12, 12, 3);
    ctx.fillStyle = '#7a90b8';
    ctx.fillText('val', W - 84, ly + 16);
  }
}

/* ─────────────────────────────────────────────────
   SECTION G — UTILITIES
───────────────────────────────────────────────── */

function setProgress(pct, text) {
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('progress-text').textContent = text;
  document.getElementById('progress-pct').textContent  = pct + '%';
}

function addLog(text, cls = 'info') {
  const content = document.getElementById('log-content');
  if (!content) return;
  const span = document.createElement('span');
  span.className = 'log-line ' + cls;
  span.textContent = text;
  content.appendChild(span);
  content.appendChild(document.createTextNode('\n'));
  content.scrollTop = content.scrollHeight;
}

function clearLog() {
  const c = document.getElementById('log-content');
  if (c) c.innerHTML = '';
}

/* ─────────────────────────────────────────────────
   INIT
───────────────────────────────────────────────── */

(async () => {
  await tf.ready();
  const backend = tf.getBackend();
  const version = tf.version.tfjs;
  document.getElementById('tf-status').textContent = `● TF.js v${version} · ${backend}`;
  document.getElementById('footer-tf').textContent = `TF.js v${version} · backend: ${backend}`;
})();