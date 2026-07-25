// dense matrix helpers; every matrix is an array of row arrays

// rows x cols matrix of zeros
export function zeros(rows, cols) {
  return Array.from({ length: rows }, () => new Array(cols).fill(0));
}

// length-n vector of zeros
export function zeroVec(n) {
  return new Array(n).fill(0);
}

// rows x cols matrix with ones on the main diagonal
export function identity(rows, cols) {
  const m = zeros(rows, cols);
  for (let i = 0; i < Math.min(rows, cols); i++) m[i][i] = 1;
  return m;
}

export function cloneMatrix(m) {
  return m.map((row) => [...row]);
}

// standard matmul; a is n x k, b is k x m
export function matmul(a, b) {
  const n = a.length;
  const k = b.length;
  const m = k === 0 ? 0 : b[0].length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) {
    const aRow = a[i];
    const outRow = out[i];
    for (let p = 0; p < k; p++) {
      const av = aRow[p];
      if (av === 0) continue;
      const bRow = b[p];
      for (let j = 0; j < m; j++) outRow[j] += av * bRow[j];
    }
  }
  return out;
}

// elementwise sum of two equally shaped matrices
export function addMatrix(a, b) {
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

// add a row vector to every row
export function addRowVec(m, v) {
  return m.map((row) => row.map((x, j) => x + v[j]));
}

export function transpose(m) {
  if (m.length === 0) return [];
  return m[0].map((_, j) => m.map((row) => row[j]));
}

export function reluMatrix(m) {
  return m.map((row) => row.map((v) => (v > 0 ? v : 0)));
}

// numerically stable row softmax; +inf entries split all the mass, a fully masked row attends nowhere
export function softmaxRow(v) {
  let max = -Infinity;
  for (const x of v) if (x > max) max = x;
  if (max === Infinity) {
    const count = v.reduce((acc, x) => acc + (x === Infinity ? 1 : 0), 0);
    return v.map((x) => (x === Infinity ? 1 / count : 0));
  }
  if (max === -Infinity) return v.map(() => 0);
  const exps = v.map((x) => Math.exp(x - max));
  const sum = exps.reduce((acc, x) => acc + x, 0);
  if (!(sum > 0)) return v.map(() => 1 / v.length);
  return exps.map((x) => x / sum);
}

export function softmaxRows(m) {
  return m.map(softmaxRow);
}

// grow/shrink a matrix, keeping the overlapping block and zero-filling the rest
export function resizeMatrix(m, rows, cols) {
  const out = zeros(rows, cols);
  for (let i = 0; i < Math.min(rows, m.length); i++) {
    for (let j = 0; j < Math.min(cols, m[i].length); j++) out[i][j] = m[i][j];
  }
  return out;
}

// grow/shrink a vector, keeping the overlapping prefix
export function resizeVec(v, n) {
  const out = zeroVec(n);
  for (let i = 0; i < Math.min(n, v.length); i++) out[i] = v[i];
  return out;
}

// slice the leading rows x cols block, used to apply a T-length prefix of a matrix
export function sliceBlock(m, rows, cols) {
  return m.slice(0, rows).map((row) => row.slice(0, cols));
}
