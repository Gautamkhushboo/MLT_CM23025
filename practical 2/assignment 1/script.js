/* Assignment 1 — Train CNN, Report Accuracy After 5 Epochs — script.js */
'use strict';

const EPOCHS  = 5;   // FIXED — Assignment requirement
const BATCH   = 128;
const PIXELS  = 784, IMG = 28, CLS = 10;
const IMGS_URL   = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

let model = null;
const epochResults = [];
const accTrain = [], accVal = [];

/* ── Load MNIST ── */
async function loadMNIST(n) {
  setP(8, 'Downloading MNIST image sprite...');
  return new Promise((res, rej) => {
    const img = new Image(); img.crossOrigin = 'anonymous';
    img.onload = async () => {
      setP(32, 'Decoding image data...');
      const c = document.createElement('canvas');
      c.width = img.width; c.height = img.height;
      c.getContext('2d').drawImage(img, 0, 0);
      const px = c.getContext('2d').getImageData(0, 0, img.width, img.height).data;
      const f32 = new Float32Array(65000 * PIXELS);
      for (let i = 0; i < f32.length; i++) f32[i] = px[i * 4] / 255;
      setP(55, 'Fetching labels...');
      try {
        const r = await fetch(LABELS_URL);
        const L = new Uint8Array(await r.arrayBuffer());
        const nt = 1500;
        const trainX = tf.tensor4d(f32.slice(0, n * PIXELS), [n, IMG, IMG, 1]);
        const trainY = tf.oneHot(tf.tensor1d(Array.from(L.slice(0, n)), 'int32'), CLS);
        const testX  = tf.tensor4d(f32.slice(55000 * PIXELS, (55000 + nt) * PIXELS), [nt, IMG, IMG, 1]);
        const testY  = tf.tensor1d(Array.from(L.slice(55000, 55000 + nt)), 'int32');
        setP(72, 'Data ready!');
        res({ trainX, trainY, testX, testY });
      } catch (e) { rej(e); }
    };
    img.onerror = () => rej(new Error('Image load failed'));
    img.src = IMGS_URL;
  });
}

/* ── Build CNN ── */
function buildCNN() {
  const m = tf.sequential();
  m.add(tf.layers.conv2d({ inputShape:[IMG,IMG,1], filters:8, kernelSize:3, activation:'relu', padding:'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize:[2,2] }));
  m.add(tf.layers.conv2d({ filters:16, kernelSize:3, activation:'relu', padding:'same' }));
  m.add(tf.layers.maxPooling2d({ poolSize:[2,2] }));
  m.add(tf.layers.flatten());
  m.add(tf.layers.dense({ units:CLS, activation:'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy'] });
  return m;
}

/* ── MAIN: Start Training ── */
async function startA1() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true; btn.textContent = '⏳ Training 5 Epochs...';

  show('prog-wrap');
  show('epoch-table-wrap');
  document.getElementById('report-card').style.display = 'none';
  epochResults.length = accTrain.length = accVal.length = 0;
  document.getElementById('et-body').innerHTML = '';

  try {
    await tf.ready();
    const n = parseInt(document.getElementById('cfg-n').value);
    const { trainX, trainY, testX, testY } = await loadMNIST(n);

    model = buildCNN();
    setP(76, 'Training CNN for 5 epochs...');

    await model.fit(trainX, trainY, {
      epochs: EPOCHS,
      batchSize: BATCH,
      validationSplit: 0.1,
      shuffle: true,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          const e    = ep + 1;
          const loss = logs.loss.toFixed(4);
          const acc  = (logs.acc * 100).toFixed(2);
          const vl   = logs.val_loss ? logs.val_loss.toFixed(4) : '—';
          const va   = logs.val_acc  ? (logs.val_acc * 100).toFixed(2) : '—';

          accTrain.push(parseFloat(acc));
          if (logs.val_acc) accVal.push(logs.val_acc * 100);

          epochResults.push({ e, loss, acc, vl, va });
          addEpochRow(e, loss, acc, vl, va);

          setP(76 + Math.round(e / EPOCHS * 24), `Epoch ${e}/${EPOCHS} done`);
          drawAccChart('chart-main', accTrain, accVal);
          await tf.nextFrame();
        },
        onTrainEnd: async () => {
          setP(100, '✔ Training complete (5 epochs)!');

          // Evaluate on test
          const ev = model.evaluate(testX, tf.oneHot(testY, CLS));
          const testLoss = (await ev[0].data())[0];
          const testAcc  = (await ev[1].data())[0];
          ev.forEach(t => t.dispose());

          showReport(testAcc, testLoss);
          btn.disabled = false; btn.textContent = '▶ Re-train';
        }
      }
    });

    trainX.dispose(); trainY.dispose(); testX.dispose(); testY.dispose();
  } catch (err) {
    setP(0, '❌ Error: ' + err.message);
    btn.disabled = false; btn.textContent = '▶ Start Training (5 Epochs)';
    console.error(err);
  }
}

/* ── Add row to epoch table ── */
function addEpochRow(e, loss, acc, vl, va) {
  const body = document.getElementById('et-body');
  const row = document.createElement('div');
  row.className = 'et-row';
  row.style.animationDelay = (e * 0.05) + 's';

  const barW = Math.min(parseFloat(acc), 100);
  row.innerHTML = `
    <span class="ep-num">${e}</span>
    <span class="ep-loss">${loss}</span>
    <span class="ep-acc">${acc}%</span>
    <span class="ep-vloss">${vl}</span>
    <span class="ep-vacc">${va !== '—' ? va + '%' : '—'}</span>
    <span class="ep-bar"><div class="acc-mini-bar" style="width:${barW}%; max-width:80px"></div></span>
  `;
  body.appendChild(row);
  body.scrollTop = body.scrollHeight;
}

/* ── Show final report ── */
function showReport(testAcc, testLoss) {
  const finalAcc  = accTrain[accTrain.length - 1];
  const finalVAcc = accVal.length ? accVal[accVal.length - 1] : null;
  const finalLoss = epochResults[epochResults.length - 1]?.loss || '—';

  document.getElementById('r-train-acc').textContent = finalAcc.toFixed(2) + '%';
  document.getElementById('r-val-acc').textContent   = finalVAcc ? finalVAcc.toFixed(2) + '%' : '—';
  document.getElementById('r-test-acc').textContent  = (testAcc * 100).toFixed(2) + '%';
  document.getElementById('r-train-loss').textContent = finalLoss;

  // Verdict
  const ta = testAcc * 100;
  let verdict = '';
  if      (ta >= 96) verdict = `<strong>Excellent!</strong> The CNN achieved ${ta.toFixed(1)}% test accuracy after 5 epochs. The model is classifying digits very accurately. Convolutional layers successfully learned spatial features from the MNIST digits.`;
  else if (ta >= 90) verdict = `<strong>Good result.</strong> The CNN achieved ${ta.toFixed(1)}% test accuracy after 5 epochs. The model is performing well. More epochs or data could push accuracy higher.`;
  else               verdict = `<strong>Moderate result.</strong> The CNN achieved ${ta.toFixed(1)}% test accuracy after 5 epochs. Consider increasing training samples or epochs for better performance.`;

  document.getElementById('verdict-text').innerHTML = verdict;
  drawAccChart('chart-main', accTrain, accVal);
  show('report-card');
}

/* ── Chart ── */
function drawAccChart(id, d1, d2, c1='#22c55e', c2='#60a5fa') {
  const cv = document.getElementById(id); if (!cv || !d1.length) return;
  const ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
  const pd = {t:14, r:20, b:36, l:44};
  const cw = W-pd.l-pd.r, ch = H-pd.t-pd.b;
  ctx.clearRect(0, 0, W, H);

  const all = [...d1, ...d2];
  const mn = Math.max(0, Math.min(...all) - 3);
  const mx = Math.min(100, Math.max(...all) + 3);
  const rng = mx - mn || 1;
  const tx = i => pd.l + (i / (d1.length - 1 || 1)) * cw;
  const ty = v => pd.t + (1 - (v - mn) / rng) * ch;

  // Grid
  for (let i = 0; i <= 5; i++) {
    const y = pd.t + (i / 5) * ch;
    ctx.strokeStyle = 'rgba(42,38,62,.6)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pd.l, y); ctx.lineTo(pd.l + cw, y); ctx.stroke();
    ctx.fillStyle = '#3d3558'; ctx.font = '9px JetBrains Mono,monospace'; ctx.textAlign = 'right';
    ctx.fillText((mx - (i/5)*rng).toFixed(1) + '%', pd.l - 5, y + 4);
  }

  // X-axis labels
  d1.forEach((_, i) => {
    ctx.fillStyle = '#3d3558'; ctx.font = '9px JetBrains Mono,monospace'; ctx.textAlign = 'center';
    ctx.fillText('Ep ' + (i+1), tx(i), H - 8);
  });

  function line(data, col) {
    if (!data.length) return;
    ctx.beginPath(); ctx.strokeStyle = col; ctx.lineWidth = 2.5; ctx.lineJoin = 'round';
    data.forEach((v, i) => i === 0 ? ctx.moveTo(tx(i), ty(v)) : ctx.lineTo(tx(i), ty(v)));
    ctx.stroke();
    data.forEach((v, i) => {
      ctx.beginPath(); ctx.arc(tx(i), ty(v), 4, 0, Math.PI * 2);
      ctx.fillStyle = col; ctx.fill();
      // Value label
      ctx.fillStyle = col; ctx.font = 'bold 9px JetBrains Mono,monospace'; ctx.textAlign = 'center';
      ctx.fillText(v.toFixed(1) + '%', tx(i), ty(v) - 8);
    });
  }

  line(d1, c1); line(d2, c2);

  // Legend
  ctx.fillStyle = c1; ctx.fillRect(pd.l, pd.t + 2, 14, 3);
  ctx.fillStyle = '#7c6fa0'; ctx.font = '9px JetBrains Mono,monospace'; ctx.textAlign = 'left';
  ctx.fillText('Train Acc', pd.l + 18, pd.t + 6);
  if (d2.length) {
    ctx.fillStyle = c2; ctx.fillRect(pd.l + 100, pd.t + 2, 14, 3);
    ctx.fillStyle = '#7c6fa0'; ctx.fillText('Val Acc', pd.l + 118, pd.t + 6);
  }
}

/* ── Helpers ── */
function setP(pct, txt) {
  document.getElementById('prog-bar').style.width = pct + '%';
  document.getElementById('prog-txt').textContent = txt;
  document.getElementById('prog-p').textContent   = pct + '%';
}
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }

/* ── Init ── */
(async () => {
  await tf.ready();
  document.getElementById('ft').textContent = `TF.js v${tf.version.tfjs} · ${tf.getBackend()}`;
})();