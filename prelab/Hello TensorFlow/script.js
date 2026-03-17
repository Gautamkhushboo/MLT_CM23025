/* ===================================================
   TensorFlow.js Practical Lab — script.js
   All tensor operations for Main Practical + Assignments
=================================================== */

/* ──────────────────────────────────────────────
   UTILITY — Output helpers
────────────────────────────────────────────── */

/**
 * Capture tf.print() / console.log() output to a DOM element.
 * Returns a cleanup function to restore originals.
 */
function captureOutput(elementId) {
  const box = document.getElementById(elementId);
  box.innerHTML = '';
  box.classList.add('running');

  const lines = [];

  const originalLog = console.log;
  const originalWarn = console.warn;

  // Override console.log
  console.log = function (...args) {
    const msg = args
      .map(a => (typeof a === 'object' ? JSON.stringify(a, null, 2) : String(a)))
      .join(' ');
    lines.push({ text: msg, cls: 'info' });
    originalLog(...args);
  };

  // TF.js uses console.log for tensor.print()
  // We intercept it above. Return restore function.
  return {
    restore() {
      console.log = originalLog;
      console.warn = originalWarn;
      box.classList.remove('running');
      renderLines(box, lines);
    },
    push(text, cls = 'info') {
      lines.push({ text, cls });
    }
  };
}

function renderLines(box, lines) {
  if (!lines.length) {
    box.innerHTML = '<span class="output-line info">// No output</span>';
    return;
  }
  box.innerHTML = lines
    .map(l => `<span class="output-line ${l.cls}">${escapeHtml(l.text)}</span>`)
    .join('');
}

function escapeHtml(str) {
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

/**
 * Safely run an async fn and show errors in output box.
 */
async function safeRun(elementId, fn) {
  const box = document.getElementById(elementId);
  try {
    await fn();
  } catch (err) {
    box.classList.remove('running');
    box.innerHTML = `<span class="output-line error">❌ Error: ${escapeHtml(err.message)}</span>`;
  }
}

/**
 * Wait for TF.js to fully load (backend ready).
 */
async function tfReady() {
  await tf.ready();
}


/* ──────────────────────────────────────────────
   MAIN PRACTICAL
────────────────────────────────────────────── */

/** STEP 01 — Create tensors and print */
async function runMain1() {
  await safeRun('main1-output', async () => {
    await tfReady();
    const cap = captureOutput('main1-output');

    cap.push('═══ Hello TensorFlow.js ═══', 'header');
    cap.push('');

    // Capture tf prints
    const original = console.log;
    const captured = [];
    console.log = (...args) => {
      captured.push(args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '));
      original(...args);
    };

    cap.push('→ Creating scalar (42):', 'label');
    const scalar = tf.scalar(42);
    scalar.print();

    cap.push('→ Creating 1D vector [1, 2, 3, 4]:', 'label');
    const vector = tf.tensor1d([1, 2, 3, 4]);
    vector.print();

    console.log = original;

    // Flush captured tf.print lines
    captured.forEach(line => cap.push('  ' + line, 'value'));

    cap.push('');
    cap.push('✔ Tensors created successfully!', 'header');

    // Cleanup
    scalar.dispose();
    vector.dispose();

    cap.restore();
  });
}

/** STEP 02 — Tensor Addition */
async function runMain2() {
  await safeRun('main2-output', async () => {
    await tfReady();

    const box = document.getElementById('main2-output');
    box.classList.add('running');
    box.innerHTML = '';

    const a = tf.tensor1d([10, 20, 30]);
    const b = tf.tensor1d([5, 8, 12]);
    const sum = tf.add(a, b);

    const aData  = await a.data();
    const bData  = await b.data();
    const sData  = await sum.data();

    const lines = [];
    lines.push({ text: '═══ Element-wise Addition ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Tensor A:', cls: 'label' });
    lines.push({ text: '  [' + Array.from(aData).join(', ') + ']', cls: 'value' });
    lines.push({ text: 'Tensor B:', cls: 'label' });
    lines.push({ text: '  [' + Array.from(bData).join(', ') + ']', cls: 'value' });
    lines.push({ text: '───────────────────', cls: 'sep' });
    lines.push({ text: 'Result (A + B):', cls: 'label' });
    lines.push({ text: '  [' + Array.from(sData).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '✔ Addition complete.', cls: 'header' });

    box.classList.remove('running');
    renderLines(box, lines);

    a.dispose(); b.dispose(); sum.dispose();
  });
}

/** STEP 03 — Verify Installation */
async function runMain3() {
  await safeRun('main3-output', async () => {
    await tfReady();

    const box = document.getElementById('main3-output');
    box.classList.add('running');
    box.innerHTML = '';

    const version = tf.version.tfjs;
    const backend  = tf.getBackend();
    const memory   = tf.memory();

    const lines = [];
    lines.push({ text: '═══ TensorFlow.js Installation Check ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '✔ TF.js Version : ' + version, cls: 'value' });
    lines.push({ text: '✔ Backend       : ' + backend, cls: 'value' });
    lines.push({ text: '✔ NumTensors    : ' + memory.numTensors, cls: 'value' });
    lines.push({ text: '✔ NumBytes      : ' + memory.numBytes, cls: 'value' });
    lines.push({ text: '✔ UnreliableIsNaN: ' + (memory.unreliable || false), cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '● TensorFlow.js is installed and running correctly!', cls: 'header' });

    box.classList.remove('running');
    renderLines(box, lines);
  });
}


/* ──────────────────────────────────────────────
   ASSIGNMENT 1 — Scalar, Vector, Matrix
────────────────────────────────────────────── */

/** A1-1: Scalar */
async function runA1_1() {
  await safeRun('a1-1-output', async () => {
    await tfReady();

    const box = document.getElementById('a1-1-output');
    box.classList.add('running');

    const scalar = tf.scalar(7);
    const data   = await scalar.data();

    const lines = [];
    lines.push({ text: '═══ Scalar Tensor (0D) ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Shape  : ' + JSON.stringify(scalar.shape), cls: 'value' });
    lines.push({ text: 'Rank   : ' + scalar.rank, cls: 'value' });
    lines.push({ text: 'DType  : ' + scalar.dtype, cls: 'value' });
    lines.push({ text: 'Value  : ' + data[0], cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '→ A scalar has 0 dimensions. Just a single number.', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    scalar.dispose();
  });
}

/** A1-2: Vector */
async function runA1_2() {
  await safeRun('a1-2-output', async () => {
    await tfReady();

    const box = document.getElementById('a1-2-output');
    box.classList.add('running');

    const vector = tf.tensor1d([1, 2, 3, 4, 5]);
    const data   = await vector.data();

    const lines = [];
    lines.push({ text: '═══ Vector Tensor (1D) ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Shape  : [' + vector.shape + ']', cls: 'value' });
    lines.push({ text: 'Rank   : ' + vector.rank, cls: 'value' });
    lines.push({ text: 'DType  : ' + vector.dtype, cls: 'value' });
    lines.push({ text: 'Values : [' + Array.from(data).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '→ A vector has 1 dimension. An ordered list of numbers.', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    vector.dispose();
  });
}

/** A1-3: Matrix */
async function runA1_3() {
  await safeRun('a1-3-output', async () => {
    await tfReady();

    const box = document.getElementById('a1-3-output');
    box.classList.add('running');

    const matrix = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const data   = await matrix.data();

    const lines = [];
    lines.push({ text: '═══ Matrix Tensor (2D) ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Shape  : [' + matrix.shape + ']  (2 rows × 3 cols)', cls: 'value' });
    lines.push({ text: 'Rank   : ' + matrix.rank, cls: 'value' });
    lines.push({ text: 'DType  : ' + matrix.dtype, cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Row 0  : [' + Array.from(data).slice(0, 3).join(', ') + ']', cls: 'value' });
    lines.push({ text: 'Row 1  : [' + Array.from(data).slice(3, 6).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '→ A matrix has 2 dimensions: rows and columns.', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    matrix.dispose();
  });
}


/* ──────────────────────────────────────────────
   ASSIGNMENT 2 — Element-wise ops on vectors
────────────────────────────────────────────── */

/** A2-1: Element-wise Addition */
async function runA2_1() {
  await safeRun('a2-1-output', async () => {
    await tfReady();

    const box = document.getElementById('a2-1-output');
    box.classList.add('running');

    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const b = tf.tensor1d([10, 20, 30, 40, 50]);
    const added = tf.add(a, b);

    const aD = Array.from(await a.data());
    const bD = Array.from(await b.data());
    const rD = Array.from(await added.data());

    const lines = [];
    lines.push({ text: '═══ Element-wise Addition ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Vector A    : [' + aD.join(', ') + ']', cls: 'label' });
    lines.push({ text: 'Vector B    : [' + bD.join(', ') + ']', cls: 'label' });
    lines.push({ text: '────────────────────────────', cls: 'sep' });
    lines.push({ text: 'A + B       : [' + rD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Each element: A[i] + B[i]', cls: 'info' });
    rD.forEach((v, i) => {
      lines.push({ text: `  [${i}]: ${aD[i]} + ${bD[i]} = ${v}`, cls: 'info' });
    });

    box.classList.remove('running');
    renderLines(box, lines);
    a.dispose(); b.dispose(); added.dispose();
  });
}

/** A2-2: Element-wise Multiplication */
async function runA2_2() {
  await safeRun('a2-2-output', async () => {
    await tfReady();

    const box = document.getElementById('a2-2-output');
    box.classList.add('running');

    const a = tf.tensor1d([1, 2, 3, 4, 5]);
    const b = tf.tensor1d([10, 20, 30, 40, 50]);
    const multiplied = tf.mul(a, b);

    const aD = Array.from(await a.data());
    const bD = Array.from(await b.data());
    const rD = Array.from(await multiplied.data());

    const lines = [];
    lines.push({ text: '═══ Element-wise Multiplication ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Vector A    : [' + aD.join(', ') + ']', cls: 'label' });
    lines.push({ text: 'Vector B    : [' + bD.join(', ') + ']', cls: 'label' });
    lines.push({ text: '────────────────────────────', cls: 'sep' });
    lines.push({ text: 'A × B       : [' + rD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Each element: A[i] × B[i]', cls: 'info' });
    rD.forEach((v, i) => {
      lines.push({ text: `  [${i}]: ${aD[i]} × ${bD[i]} = ${v}`, cls: 'info' });
    });

    box.classList.remove('running');
    renderLines(box, lines);
    a.dispose(); b.dispose(); multiplied.dispose();
  });
}

/** A2-3: Both Together */
async function runA2_3() {
  await safeRun('a2-3-output', async () => {
    await tfReady();

    const box = document.getElementById('a2-3-output');
    box.classList.add('running');

    const a = tf.tensor1d([2, 4, 6]);
    const b = tf.tensor1d([3, 3, 3]);

    const added = tf.add(a, b);
    const muled  = tf.mul(a, b);

    const aD  = Array.from(await a.data());
    const bD  = Array.from(await b.data());
    const adD = Array.from(await added.data());
    const muD = Array.from(await muled.data());

    const lines = [];
    lines.push({ text: '═══ Element-wise Ops Summary ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Vector A         : [' + aD.join(', ') + ']', cls: 'label' });
    lines.push({ text: 'Vector B         : [' + bD.join(', ') + ']', cls: 'label' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'tf.add(A, B)     : [' + adD.join(', ') + ']', cls: 'value' });
    lines.push({ text: 'tf.mul(A, B)     : [' + muD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '✔ add() uses +  |  mul() uses ×', cls: 'header' });

    box.classList.remove('running');
    renderLines(box, lines);
    a.dispose(); b.dispose(); added.dispose(); muled.dispose();
  });
}


/* ──────────────────────────────────────────────
   ASSIGNMENT 3 — reshape() vs flatten()
────────────────────────────────────────────── */

/** A3-1: reshape() */
async function runA3_1() {
  await safeRun('a3-1-output', async () => {
    await tfReady();

    const box = document.getElementById('a3-1-output');
    box.classList.add('running');

    const t        = tf.tensor1d([1, 2, 3, 4, 5, 6]);
    const reshaped = t.reshape([2, 3]);
    const tData    = Array.from(await t.data());
    const rData    = Array.from(await reshaped.data());

    const lines = [];
    lines.push({ text: '═══ reshape() Demo ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Original tensor [6]:', cls: 'label' });
    lines.push({ text: '  Shape : [' + t.shape + ']  Rank: ' + t.rank, cls: 'value' });
    lines.push({ text: '  Data  : [' + tData.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'After .reshape([2, 3]):', cls: 'label' });
    lines.push({ text: '  Shape : [' + reshaped.shape + ']  Rank: ' + reshaped.rank, cls: 'value' });
    lines.push({ text: '  Row 0 : [' + rData.slice(0,3).join(', ') + ']', cls: 'value' });
    lines.push({ text: '  Row 1 : [' + rData.slice(3,6).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '→ reshape() changes shape but NOT the data.', cls: 'info' });
    lines.push({ text: '→ Total elements must remain the same (6 = 2×3).', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    t.dispose(); reshaped.dispose();
  });
}

/** A3-2: flatten() */
async function runA3_2() {
  await safeRun('a3-2-output', async () => {
    await tfReady();

    const box = document.getElementById('a3-2-output');
    box.classList.add('running');

    const mat  = tf.tensor2d([[1, 2, 3], [4, 5, 6]]);
    const flat = mat.flatten();
    const mD   = Array.from(await mat.data());
    const fD   = Array.from(await flat.data());

    const lines = [];
    lines.push({ text: '═══ flatten() Demo ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Original matrix [2,3]:', cls: 'label' });
    lines.push({ text: '  Shape : [' + mat.shape + ']  Rank: ' + mat.rank, cls: 'value' });
    lines.push({ text: '  Row 0 : [' + mD.slice(0,3).join(', ') + ']', cls: 'value' });
    lines.push({ text: '  Row 1 : [' + mD.slice(3,6).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'After .flatten():', cls: 'label' });
    lines.push({ text: '  Shape : [' + flat.shape + ']  Rank: ' + flat.rank, cls: 'value' });
    lines.push({ text: '  Data  : [' + fD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '→ flatten() always outputs a 1D tensor.', cls: 'info' });
    lines.push({ text: '→ Works on ANY tensor shape, no arguments needed.', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    mat.dispose(); flat.dispose();
  });
}

/** A3-3: Comparison */
async function runA3_3() {
  await safeRun('a3-3-output', async () => {
    await tfReady();

    const box = document.getElementById('a3-3-output');
    box.classList.add('running');

    // 3D tensor [2,2,2]
    const t = tf.tensor3d([[[1,2],[3,4]],[[5,6],[7,8]]]);
    const r = t.reshape([4, 2]);
    const f = t.flatten();

    const tD = Array.from(await t.data());
    const rD = Array.from(await r.data());
    const fD = Array.from(await f.data());

    const lines = [];
    lines.push({ text: '═══ reshape() vs flatten() ═══', cls: 'header' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'Original 3D tensor [2,2,2]:', cls: 'label' });
    lines.push({ text: '  Shape: [' + t.shape + ']  Rank: ' + t.rank, cls: 'value' });
    lines.push({ text: '  All values: [' + tD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '── reshape([4,2]) ──', cls: 'label' });
    lines.push({ text: '  Shape: [' + r.shape + ']  Rank: ' + r.rank, cls: 'value' });
    lines.push({ text: '  Row 0: [' + rD.slice(0,2).join(', ') + ']', cls: 'value' });
    lines.push({ text: '  Row 1: [' + rD.slice(2,4).join(', ') + ']', cls: 'value' });
    lines.push({ text: '  Row 2: [' + rD.slice(4,6).join(', ') + ']', cls: 'value' });
    lines.push({ text: '  Row 3: [' + rD.slice(6,8).join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: '── flatten() ──', cls: 'label' });
    lines.push({ text: '  Shape: [' + f.shape + ']  Rank: ' + f.rank, cls: 'value' });
    lines.push({ text: '  Data : [' + fD.join(', ') + ']', cls: 'value' });
    lines.push({ text: '', cls: 'info' });
    lines.push({ text: 'KEY DIFFERENCE:', cls: 'header' });
    lines.push({ text: '  reshape() → you choose the output shape (must be valid)', cls: 'info' });
    lines.push({ text: '  flatten() → always gives a 1D [n] vector, no choice', cls: 'info' });

    box.classList.remove('running');
    renderLines(box, lines);
    t.dispose(); r.dispose(); f.dispose();
  });
}


/* ──────────────────────────────────────────────
   TAB NAVIGATION
────────────────────────────────────────────── */

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;

    // Update buttons
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Update panels
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.getElementById('tab-' + target).classList.add('active');
  });
});


/* ──────────────────────────────────────────────
   INIT — Show TF version in footer
────────────────────────────────────────────── */

(async () => {
  await tf.ready();
  const version = tf.version.tfjs;
  const backend  = tf.getBackend();
  const vEl = document.getElementById('tf-version-footer');
  if (vEl) vEl.textContent = `TF.js v${version} · Backend: ${backend}`;
})();