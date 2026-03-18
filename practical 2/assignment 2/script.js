/* Assignment 2 — Draw a digit and classify using trained CNN — script.js */
'use strict';

const PIXELS = 784, IMG = 28, CLS = 10;
const IMGS_URL   = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const LABELS_URL = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

let model = null;
const predHistory = [];

/* ══════════════════════════════════════════
   SECTION 1 — TRAIN CNN
══════════════════════════════════════════ */

async function doTrain() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true; btn.textContent = '⏳ Training...';
  show('train-prog');
  document.getElementById('train-done-box').style.display = 'none';
  setStep(1);

  try {
    await tf.ready();
    const n = parseInt(document.getElementById('cfg-n').value);

    // Load data
    setP(8, 'Downloading MNIST sprite...');
    const { trainX, trainY } = await loadMNIST(n);

    // Build model
    model = tf.sequential();
    model.add(tf.layers.conv2d({ inputShape:[IMG,IMG,1], filters:8, kernelSize:3, activation:'relu', padding:'same' }));
    model.add(tf.layers.maxPooling2d({ poolSize:[2,2] }));
    model.add(tf.layers.conv2d({ filters:16, kernelSize:3, activation:'relu', padding:'same' }));
    model.add(tf.layers.maxPooling2d({ poolSize:[2,2] }));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({ units:CLS, activation:'softmax' }));
    model.compile({ optimizer: tf.train.adam(0.001), loss:'categoricalCrossentropy', metrics:['accuracy'] });

    setP(76, 'Training CNN...');
    let finalAcc = 0, finalLoss = 0;

    await model.fit(trainX, trainY, {
      epochs: 5, batchSize: 128,
      validationSplit: 0.1, shuffle: true,
      callbacks: {
        onEpochEnd: async (ep, logs) => {
          finalAcc  = (logs.acc * 100).toFixed(1);
          finalLoss = logs.loss.toFixed(4);
          const pct = 76 + Math.round((ep+1)/5*24);
          setP(pct, `Epoch ${ep+1}/5 — acc: ${finalAcc}%  loss: ${finalLoss}`);
          document.getElementById('tp-stats').textContent = `Epoch ${ep+1}/5 | accuracy: ${finalAcc}% | loss: ${finalLoss}`;
          await tf.nextFrame();
        },
        onTrainEnd: () => {
          setP(100, '✔ Training complete!');
          document.getElementById('tdb-sub').textContent = `Trained 5 epochs · Final accuracy: ${finalAcc}% · Loss: ${finalLoss}`;
          show('train-done-box');
          btn.textContent = '▶ Re-train';
          btn.disabled = false;

          document.getElementById('btn-predict').disabled = false;
          show('intended-row');
          buildDigitBtns();
          setStep(2);
        }
      }
    });

    trainX.dispose(); trainY.dispose();
  } catch (err) {
    setP(0, '❌ Error: ' + err.message);
    btn.textContent = '⬡ Load MNIST & Train';
    btn.disabled = false;
    console.error(err);
  }
}

/* ══════════════════════════════════════════
   SECTION 2 — DRAW CANVAS
══════════════════════════════════════════ */

const dc  = document.getElementById('draw-canvas');
const dctx = dc.getContext('2d');
let brushSize = 22, down = false, lx = 0, ly = 0;

dctx.fillStyle = '#000';
dctx.fillRect(0, 0, 280, 280);
dctx.strokeStyle = '#fff';
dctx.lineWidth   = brushSize;
dctx.lineCap = dctx.lineJoin = 'round';

function updateBrush(val) {
  brushSize = parseInt(val);
  dctx.lineWidth = brushSize;
  document.getElementById('brush-val').textContent = val + 'px';
}

function getPos(e) {
  const r  = dc.getBoundingClientRect();
  const sx = dc.width / r.width, sy = dc.height / r.height;
  if (e.touches) return { x:(e.touches[0].clientX-r.left)*sx, y:(e.touches[0].clientY-r.top)*sy };
  return { x:(e.clientX-r.left)*sx, y:(e.clientY-r.top)*sy };
}

dc.addEventListener('mousedown', e => { down=true; const p=getPos(e); lx=p.x; ly=p.y; dctx.beginPath(); dctx.arc(p.x,p.y,brushSize/2,0,Math.PI*2); dctx.fillStyle='#fff'; dctx.fill(); document.getElementById('canvas-lbl').style.opacity='0'; });
dc.addEventListener('mousemove', e => { if (!down) return; const p=getPos(e); dctx.beginPath(); dctx.moveTo(lx,ly); dctx.lineTo(p.x,p.y); dctx.stroke(); lx=p.x; ly=p.y; if (model) { clearTimeout(dc._t); dc._t=setTimeout(classify,200); } });
dc.addEventListener('mouseup',    () => down=false);
dc.addEventListener('mouseleave', () => down=false);
dc.addEventListener('touchstart', e => { e.preventDefault(); down=true; const p=getPos(e); lx=p.x; ly=p.y; document.getElementById('canvas-lbl').style.opacity='0'; }, {passive:false});
dc.addEventListener('touchmove',  e => { e.preventDefault(); if (!down) return; const p=getPos(e); dctx.beginPath(); dctx.moveTo(lx,ly); dctx.lineTo(p.x,p.y); dctx.stroke(); lx=p.x; ly=p.y; if (model) { clearTimeout(dc._t); dc._t=setTimeout(classify,200); } }, {passive:false});
dc.addEventListener('touchend', () => down=false);

function clearDraw() {
  dctx.fillStyle = '#000'; dctx.fillRect(0, 0, 280, 280);
  document.getElementById('canvas-lbl').style.opacity = '1';
  document.getElementById('pred-waiting').style.display = 'flex';
  document.getElementById('pred-result').style.display  = 'none';
  resetDigitBtns();
}

/* ══════════════════════════════════════════
   SECTION 3 — CLASSIFY
══════════════════════════════════════════ */

async function classify() {
  if (!model) return;
  setStep(3);

  document.getElementById('pred-waiting').style.display = 'none';
  document.getElementById('pred-result').style.display  = 'flex';

  const t = tf.tidy(() => {
    const s = document.createElement('canvas'); s.width = s.height = 28;
    const sx = s.getContext('2d'); sx.imageSmoothingEnabled = true;
    sx.drawImage(dc, 0, 0, 28, 28);
    const id  = sx.getImageData(0, 0, 28, 28);
    const inp = new Float32Array(784);
    for (let i = 0; i < 784; i++) inp[i] = id.data[i*4] / 255;
    return tf.tensor4d(inp, [1,28,28,1]);
  });

  const pt = model.predict(t); t.dispose();
  const probs = Array.from(await pt.data()); pt.dispose();
  const best  = probs.indexOf(Math.max(...probs));
  const conf  = (probs[best] * 100).toFixed(1);

  document.getElementById('pr-digit').textContent = best;
  document.getElementById('pr-conf').textContent  = conf + '%';

  // Probability bars
  const bars = document.getElementById('ap-bars'); bars.innerHTML = '';
  probs.forEach((v, d) => {
    const pct = (v * 100).toFixed(1);
    bars.innerHTML += `<div class="apb-row"><div class="apb-d">${d}</div><div class="apb-bg"><div class="apb-fill ${d===best?'top':''}" style="width:${pct}%"></div></div><div class="apb-pct">${pct}%</div></div>`;
  });

  // Add to history
  addHistory(best, conf);
  resetDigitBtns(best);
}

/* ── History ── */
function addHistory(digit, conf) {
  predHistory.unshift({ digit, conf });
  if (predHistory.length > 8) predHistory.pop();
  const list = document.getElementById('hb-list'); list.innerHTML = '';
  predHistory.forEach(h => {
    list.innerHTML += `<div class="hb-item"><div class="hb-d">${h.digit}</div><div class="hb-c">${h.conf}%</div></div>`;
  });
  show('history-box');
}

/* ── Intended-digit feedback buttons ── */
function buildDigitBtns() {
  const wrap = document.getElementById('digit-btns'); wrap.innerHTML = '';
  for (let d = 0; d < 10; d++) {
    const b = document.createElement('button');
    b.className = 'digit-btn'; b.textContent = d; b.dataset.d = d;
    b.onclick = () => checkFeedback(d);
    wrap.appendChild(b);
  }
}
function resetDigitBtns(predicted) {
  document.querySelectorAll('.digit-btn').forEach(b => {
    b.classList.remove('correct', 'wrong');
    if (predicted !== undefined && parseInt(b.dataset.d) === predicted) b.classList.add('correct');
  });
}
function checkFeedback(intended) {
  const predDigit = parseInt(document.getElementById('pr-digit').textContent);
  document.querySelectorAll('.digit-btn').forEach(b => {
    b.classList.remove('correct', 'wrong');
    const d = parseInt(b.dataset.d);
    if (d === intended && d === predDigit) b.classList.add('correct');
    else if (d === intended && d !== predDigit) b.classList.add('wrong');
    else if (d === predDigit && d !== intended) b.classList.add('wrong');
  });
}

/* ══════════════════════════════════════════
   DATA LOADING
══════════════════════════════════════════ */

async function loadMNIST(n) {
  return new Promise((res, rej) => {
    const img = new Image(); img.crossOrigin='anonymous';
    img.onload = async () => {
      setP(34, 'Decoding pixels...');
      const c=document.createElement('canvas'); c.width=img.width; c.height=img.height;
      c.getContext('2d').drawImage(img,0,0);
      const px=c.getContext('2d').getImageData(0,0,img.width,img.height).data;
      const f32=new Float32Array(65000*PIXELS);
      for (let i=0;i<f32.length;i++) f32[i]=px[i*4]/255;
      setP(55,'Fetching labels...');
      try {
        const r=await fetch(LABELS_URL);
        const L=new Uint8Array(await r.arrayBuffer());
        const trainX=tf.tensor4d(f32.slice(0,n*PIXELS),[n,IMG,IMG,1]);
        const trainY=tf.oneHot(tf.tensor1d(Array.from(L.slice(0,n)),'int32'),CLS);
        setP(74,'Data ready!'); res({ trainX, trainY });
      } catch(e){rej(e);}
    };
    img.onerror=()=>rej(new Error('Sprite load failed'));
    img.src=IMGS_URL;
  });
}

/* ══════════════════════════════════════════
   HELPERS
══════════════════════════════════════════ */

function setP(pct, txt) {
  document.getElementById('tp-fill').style.width = pct + '%';
  document.getElementById('tp-txt').textContent  = txt;
}
function show(id) { const e=document.getElementById(id); if(e) e.style.display=''; }

function setStep(active) {
  [1,2,3].forEach(n => {
    const el = document.getElementById('step-' + n);
    if (!el) return;
    el.classList.remove('done','active');
    if (n < active) el.classList.add('done');
    else if (n === active) el.classList.add('active');
  });
}

/* ── Init ── */
(async () => {
  await tf.ready();
  document.getElementById('ft').textContent = `TF.js v${tf.version.tfjs} · ${tf.getBackend()}`;
})();