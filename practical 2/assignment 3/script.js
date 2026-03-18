/* Assignment 3 — CNN vs Dense Network Comparison — script.js */
'use strict';

const PIXELS = 784, IMG = 28, CLS = 10;
const IMGS_URL   = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

// Results storage
const results = {
  cnn:   { accH:[], valAccH:[], lossH:[], valLossH:[], testAcc:0, testLoss:0, params:0, time:0 },
  dense: { accH:[], valAccH:[], lossH:[], valLossH:[], testAcc:0, testLoss:0, params:0, time:0 }
};

/* ══════════════════════════════════════════════
   MAIN: Run Comparison
══════════════════════════════════════════════ */

async function runComparison() {
  const btn = document.getElementById('btn-run');
  btn.disabled = true; btn.textContent = '⏳ Training Both Models...';

  show('live-section');
  document.getElementById('results-section').style.display = 'none';

  // Reset
  Object.keys(results).forEach(k => {
    const r = results[k];
    r.accH=[]; r.valAccH=[]; r.lossH=[]; r.valLossH=[];
    r.testAcc=0; r.testLoss=0; r.params=0; r.time=0;
  });

  try {
    await tf.ready();
    const n = parseInt(document.getElementById('cfg-n').value);

    setStatus('cnn',   'Loading data...');
    setStatus('dense', 'Waiting...');

    const { trainX, trainY, testX, testY } = await loadData(n);

    // ── Train CNN first ──
    setStatus('cnn', 'Training...');
    const cnnModel = buildCNN();
    document.getElementById('cnn-params').textContent   = cnnModel.countParams().toLocaleString();
    document.getElementById('cnn-params-r').textContent = cnnModel.countParams().toLocaleString();
    results.cnn.params = cnnModel.countParams();

    const cnnStart = performance.now();
    await trainModel(cnnModel, trainX, trainY, 'cnn');
    results.cnn.time = ((performance.now() - cnnStart) / 1000).toFixed(1);

    // Evaluate CNN
    const cnnEv = cnnModel.evaluate(testX, tf.oneHot(testY, CLS));
    results.cnn.testLoss = (await cnnEv[0].data())[0];
    results.cnn.testAcc  = (await cnnEv[1].data())[0];
    cnnEv.forEach(t => t.dispose());
    setStatus('cnn', '✔ Done — ' + (results.cnn.testAcc * 100).toFixed(1) + '% acc');

    // ── Train Dense next ──
    setStatus('dense', 'Training...');
    const denseModel = buildDense();
    document.getElementById('dense-params').textContent   = denseModel.countParams().toLocaleString();
    document.getElementById('dense-params-r').textContent = denseModel.countParams().toLocaleString();
    results.dense.params = denseModel.countParams();

    // Dense needs flat input
    const flatTestX = testX.reshape([testX.shape[0], PIXELS]);

    const dStart = performance.now();
    await trainModel(denseModel, trainX.reshape([trainX.shape[0], PIXELS]), trainY, 'dense');
    results.dense.time = ((performance.now() - dStart) / 1000).toFixed(1);

    // Evaluate Dense
    const denseEv = denseModel.evaluate(flatTestX, tf.oneHot(testY, CLS));
    results.dense.testLoss = (await denseEv[0].data())[0];
    results.dense.testAcc  = (await denseEv[1].data())[0];
    denseEv.forEach(t => t.dispose());
    setStatus('dense', '✔ Done — ' + (results.dense.testAcc * 100).toFixed(1) + '% acc');

    trainX.dispose(); trainY.dispose(); testX.dispose(); testY.dispose(); flatTestX.dispose();
    cnnModel.dispose(); denseModel.dispose();

    showResults();
    btn.disabled = false; btn.textContent = '⬡ Re-run Comparison';

  } catch (err) {
    console.error(err);
    setStatus('cnn', '❌ Error');
    setStatus('dense', '❌ Error');
    btn.disabled = false; btn.textContent = '⬡ Train Both & Compare';
  }
}

/* ══════════════════════════════════════════════
   BUILD MODELS
══════════════════════════════════════════════ */

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

function buildDense() {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape:[PIXELS], units:128, activation:'relu' }));
  m.add(tf.layers.dense({ units:64, activation:'relu' }));
  m.add(tf.layers.dense({ units:CLS, activation:'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy'] });
  return m;
}

/* ══════════════════════════════════════════════
   TRAIN MODEL
══════════════════════════════════════════════ */

async function trainModel(m, X, Y, key) {
  const epochs = parseInt(document.getElementById('cfg-e').value);
  const batch  = parseInt(document.getElementById('cfg-b').value);
  const r      = results[key];

  await m.fit(X, Y, {
    epochs, batchSize: batch,
    validationSplit: 0.1, shuffle: true,
    callbacks: {
      onEpochEnd: async (ep, logs) => {
        const e   = ep + 1;
        const acc = logs.acc * 100;
        const va  = logs.val_acc ? logs.val_acc * 100 : null;

        r.accH.push(acc);
        r.lossH.push(logs.loss);
        if (va !== null) r.valAccH.push(va);
        if (logs.val_loss) r.valLossH.push(logs.val_loss);

        const pct = Math.round(e / epochs * 100);
        setProg(key, pct, `Epoch ${e}/${epochs}`, `acc: ${acc.toFixed(1)}%  loss: ${logs.loss.toFixed(4)}`);
        await tf.nextFrame();
      }
    }
  });
}

/* ══════════════════════════════════════════════
   SHOW RESULTS
══════════════════════════════════════════════ */

function showResults() {
  show('results-section');

  const ca = (results.cnn.testAcc   * 100).toFixed(2) + '%';
  const da = (results.dense.testAcc * 100).toFixed(2) + '%';
  const cl = results.cnn.testLoss.toFixed(4);
  const dl = results.dense.testLoss.toFixed(4);

  document.getElementById('cnn-acc').textContent       = ca;
  document.getElementById('cnn-loss-final').textContent = cl;
  document.getElementById('cnn-time').textContent       = results.cnn.time + 's';
  document.getElementById('cnn-params-r').textContent   = results.cnn.params.toLocaleString();

  document.getElementById('dense-acc').textContent        = da;
  document.getElementById('dense-loss-final').textContent  = dl;
  document.getElementById('dense-time').textContent        = results.dense.time + 's';
  document.getElementById('dense-params-r').textContent    = results.dense.params.toLocaleString();

  const cnnBetter  = results.cnn.testAcc > results.dense.testAcc;
  const tied       = Math.abs(results.cnn.testAcc - results.dense.testAcc) < 0.005;
  const diff       = Math.abs((results.cnn.testAcc - results.dense.testAcc) * 100).toFixed(2);

  // Winner banner
  let wb = '';
  if (tied)         wb = `<strong>It's a tie!</strong> Both models achieved similar accuracy (~${ca}). Dense networks can match CNNs on simple datasets like MNIST, but CNNs generally scale better to complex vision tasks.`;
  else if (cnnBetter) wb = `<strong>CNN wins!</strong> The CNN achieved ${ca} vs Dense ${da} — a <strong>${diff}% improvement</strong>. Convolutional layers capture spatial features in images more effectively than dense layers.`;
  else               wb = `<strong>Dense wins!</strong> The Dense network achieved ${da} vs CNN ${ca} — ${diff}% better this run. For simple datasets like MNIST, a well-tuned dense network can sometimes match or beat a CNN.`;
  document.getElementById('wb-text').innerHTML = wb;

  // Comparison table rows
  fillRow('ct-acc',    'Test Accuracy',  ca, da, cnnBetter ? 'cnn' : (tied ? 'tie' : 'dense'));
  fillRow('ct-loss',   'Test Loss',      cl, dl, parseFloat(cl) < parseFloat(dl) ? 'cnn' : 'dense');
  fillRow('ct-params', 'Parameters',
    results.cnn.params.toLocaleString(),
    results.dense.params.toLocaleString(),
    results.cnn.params < results.dense.params ? 'cnn' : 'dense',
    '(fewer = simpler)'
  );
  fillRow('ct-time', 'Train Time',
    results.cnn.time + 's',
    results.dense.time + 's',
    parseFloat(results.cnn.time) < parseFloat(results.dense.time) ? 'cnn' : 'dense',
    '(faster = better)'
  );

  // Chart
  drawCompChart();

  // Analysis
  const analysis = `
    <strong>Analysis:</strong><br/><br/>
    The CNN uses convolutional layers that apply learnable filters to detect edges, curves, and shapes in images — 
    making them inherently suited for visual data. The Dense (fully-connected) network treats each pixel independently, 
    losing spatial relationships.<br/><br/>
    <strong>Key observations from this run:</strong><br/>
    • CNN Test Accuracy: <strong>${ca}</strong> &nbsp;|&nbsp; Dense Test Accuracy: <strong>${da}</strong><br/>
    • CNN Parameters: <strong>${results.cnn.params.toLocaleString()}</strong> &nbsp;|&nbsp; Dense Parameters: <strong>${results.dense.params.toLocaleString()}</strong><br/>
    • CNN Train Time: <strong>${results.cnn.time}s</strong> &nbsp;|&nbsp; Dense Train Time: <strong>${results.dense.time}s</strong><br/><br/>
    For MNIST specifically, both architectures perform well because the digit images are simple and centered. 
    However, on more complex datasets (CIFAR-10, ImageNet), CNNs dramatically outperform dense networks 
    due to their spatial feature learning and parameter efficiency through weight sharing in convolutional filters.
  `;
  document.getElementById('analysis-box').innerHTML = analysis;
}

function fillRow(id, label, cnnVal, denseVal, winner, note='') {
  const row = document.getElementById(id);
  let winHtml = '';
  if      (winner === 'cnn')   winHtml = `<span class="ct-win cnn-win">CNN${note?' '+note:''}</span>`;
  else if (winner === 'dense') winHtml = `<span class="ct-win dense-win">Dense${note?' '+note:''}</span>`;
  else                         winHtml = `<span class="ct-win tie">Tie</span>`;
  row.innerHTML = `<span>${label}</span><span class="ct-cnn">${cnnVal}</span><span class="ct-dense">${denseVal}</span>${winHtml}`;
}

/* ══════════════════════════════════════════════
   CHART
══════════════════════════════════════════════ */

function drawCompChart() {
  const cv  = document.getElementById('comp-chart'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height;
  const pd = {t:18, r:20, b:36, l:48};
  const cw = W-pd.l-pd.r, ch = H-pd.t-pd.b;
  ctx.clearRect(0, 0, W, H);

  const { accH:ca, valAccH:cva } = results.cnn;
  const { accH:da, valAccH:dva } = results.dense;
  const all = [...ca, ...cva, ...da, ...dva].filter(v => v > 0);
  if (!all.length) return;

  const mn = Math.max(0, Math.min(...all) - 3);
  const mx = Math.min(100, Math.max(...all) + 3);
  const rng = mx - mn || 1;
  const len = ca.length;
  const tx  = i => pd.l + (i / (len-1||1)) * cw;
  const ty  = v => pd.t + (1 - (v-mn)/rng) * ch;

  // Grid
  for (let i=0; i<=5; i++) {
    const y = pd.t + (i/5)*ch;
    ctx.strokeStyle='rgba(26,34,52,.7)'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(pd.l,y); ctx.lineTo(pd.l+cw,y); ctx.stroke();
    ctx.fillStyle='#2e3f58'; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='right';
    ctx.fillText((mx-(i/5)*rng).toFixed(0)+'%', pd.l-5, y+4);
  }
  ca.forEach((_,i) => {
    ctx.fillStyle='#2e3f58'; ctx.font='9px JetBrains Mono,monospace'; ctx.textAlign='center';
    ctx.fillText('Ep'+(i+1), tx(i), H-8);
  });

  function line(data, color, dashed=false) {
    if (!data.length) return;
    ctx.beginPath();
    ctx.strokeStyle=color; ctx.lineWidth=2.5; ctx.lineJoin='round';
    if (dashed) ctx.setLineDash([5,4]); else ctx.setLineDash([]);
    data.forEach((v,i) => i===0?ctx.moveTo(tx(i),ty(v)):ctx.lineTo(tx(i),ty(v)));
    ctx.stroke();
    ctx.setLineDash([]);
    data.forEach((v,i) => { ctx.beginPath(); ctx.arc(tx(i),ty(v),3.5,0,Math.PI*2); ctx.fillStyle=color; ctx.fill(); });
  }

  line(ca,  '#f97316');
  line(cva, '#fdba74', true);
  line(da,  '#818cf8');
  line(dva, '#c4b5fd', true);
}

/* ══════════════════════════════════════════════
   DATA LOADING
══════════════════════════════════════════════ */

async function loadData(n) {
  setProg('cnn', 10, 'Fetching MNIST sprite...', '');
  return new Promise((res, rej) => {
    const img = new Image(); img.crossOrigin='anonymous';
    img.onload = async () => {
      setProg('cnn', 38, 'Decoding pixels...', '');
      const c = document.createElement('canvas'); c.width=img.width; c.height=img.height;
      c.getContext('2d').drawImage(img,0,0);
      const px = c.getContext('2d').getImageData(0,0,img.width,img.height).data;
      const f32 = new Float32Array(65000*PIXELS);
      for (let i=0;i<f32.length;i++) f32[i]=px[i*4]/255;
      setProg('cnn', 56, 'Fetching labels...', '');
      try {
        const r = await fetch(LABELS_URL);
        const L = new Uint8Array(await r.arrayBuffer());
        const nt = 1500;
        const trainX = tf.tensor4d(f32.slice(0,n*PIXELS), [n,IMG,IMG,1]);
        const trainY = tf.oneHot(tf.tensor1d(Array.from(L.slice(0,n)),'int32'),CLS);
        const testX  = tf.tensor4d(f32.slice(55000*PIXELS,(55000+nt)*PIXELS),[nt,IMG,IMG,1]);
        const testY  = tf.tensor1d(Array.from(L.slice(55000,55000+nt)),'int32');
        setProg('cnn', 72, 'Data ready!', '');
        res({ trainX, trainY, testX, testY });
      } catch(e){rej(e);}
    };
    img.onerror=()=>rej(new Error('Sprite load failed'));
    img.src=IMGS_URL;
  });
}

/* ══════════════════════════════════════════════
   HELPERS
══════════════════════════════════════════════ */

function setProg(key, pct, txt, metrics) {
  const fill = document.getElementById(key+'-fill'); if(fill) fill.style.width=pct+'%';
  const ptxt = document.getElementById(key+'-prog-txt'); if(ptxt) ptxt.textContent=txt;
  const pct_ = document.getElementById(key+'-prog-p');  if(pct_) pct_.textContent=pct+'%';
  const met  = document.getElementById(key+'-metrics'); if(met && metrics) met.textContent=metrics;
}

function setStatus(key, txt) {
  const el = document.getElementById(key+'-status'); if(el) el.textContent=txt;
}

function show(id) { const e=document.getElementById(id); if(e) e.style.display=''; }

/* ── Init ── */
(async () => {
  await tf.ready();
  document.getElementById('ft').textContent = `TF.js v${tf.version.tfjs} · ${tf.getBackend()}`;
})();