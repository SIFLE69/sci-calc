// src/ScientificCalculator.jsx
import React, { useEffect, useRef, useState } from "react";
import { create, all } from "mathjs";

/*
  ScientificCalculator.jsx — FINAL
  - Matrix mode (responsive determinant + rendered matrix result)
  - Vector mode (dot, cross, scalar triple, vector triple, resultant, projection, magnitude, normalize, planes)
  - Calculus mode (diff, partial diff, numeric integrate (Simpson), definite integrals)
  - Fraction-aware display
  - Keyboard navigation, save/load matrices, CSV export
  - Dark mode support
  - Requires: npm install mathjs
*/

const math = create(all);
math.config({ number: "Fraction", precision: 64, matrix: "Array" });
const mathSci = create(all);
mathSci.config({ number: "number", precision: 64, matrix: "Array" });

// ----------------- Helpers -----------------
function gcd(a, b) {
  a = Math.abs(a) | 0; b = Math.abs(b) | 0;
  if (!b) return a;
  while (b) { [a, b] = [b, a % b]; }
  return a;
}
function toFraction(x, maxDenominator = 1000, eps = 1e-12) {
  if (!isFinite(x)) return null;
  const sign = x < 0 ? -1 : 1; x = Math.abs(x);
  if (Math.abs(x - Math.round(x)) < eps) return { n: sign * Math.round(x), d: 1 };
  let a0 = Math.floor(x), h0 = 1, k0 = 0, h1 = a0, k1 = 1, frac = x - a0, iter = 0;
  while (iter < 60 && Math.abs(h1 / k1 - x) > eps && k1 <= maxDenominator) {
    if (frac === 0) break;
    frac = 1 / frac; const ai = Math.floor(frac);
    const h2 = ai * h1 + h0, k2 = ai * k1 + k0;
    h0 = h1; k0 = k1; h1 = h2; k1 = k2; frac = frac - ai; iter++;
  }
  if (k1 > maxDenominator) {
    const d = maxDenominator, n = Math.round(x * d), g = gcd(n, d);
    return { n: sign * (n / g), d: d / g };
  }
  return { n: sign * h1, d: k1 };
}
function formatDecimal(x) {
  if (!isFinite(x)) return String(x);
  return Number.parseFloat(x.toPrecision(12)).toString();
}
function safeFormatNumberAsFractionOrDecimal(x, preferFraction = true) {
  if (!isFinite(x)) return String(x);
  if (preferFraction) {
    const frac = toFraction(x, 1000, 1e-12);
    if (frac && Math.abs(x - frac.n / frac.d) <= 1e-10 && frac.d <= 1000) {
      if (frac.d === 1) return String(frac.n);
      return `${frac.n}/${frac.d} (${formatDecimal(x)})`;
    }
  }
  return formatDecimal(x);
}
function formatResult(evalResult, preferFraction = true) {
  const type = math.typeOf ? math.typeOf(evalResult) : typeof evalResult;
  if (type === "Fraction") {
    const numeric = evalResult.valueOf();
    return `${evalResult.toString()} (${formatDecimal(numeric)})`;
  }
  if (type === "BigNumber") {
    const val = evalResult.toNumber();
    return safeFormatNumberAsFractionOrDecimal(val, preferFraction);
  }
  if (type === "Complex") {
    const re = formatDecimal(evalResult.re), im = formatDecimal(Math.abs(evalResult.im));
    return `${re} ${evalResult.im >= 0 ? "+" : "-"} ${im}i`;
  }
  if (Array.isArray(evalResult)) {
    const rows = evalResult.map(r => "[" + r.map(v => {
      if (typeof v === "number") return safeFormatNumberAsFractionOrDecimal(v, preferFraction);
      try { return math.format(v, { fraction: "ratio", precision: 12 }); } catch { return String(v); }
    }).join(", ") + "]");
    return "[" + rows.join(", ") + "]";
  }
  if (typeof evalResult === "number") return safeFormatNumberAsFractionOrDecimal(evalResult, preferFraction);
  try { return math.format(evalResult, { fraction: "ratio", precision: 12 }); } catch { return String(evalResult); }
}

// matrix adjugate
function adjointMatrix(m) {
  const size = math.size(m).valueOf();
  if (size.length !== 2 || size[0] !== size[1]) throw new Error("adj: matrix must be square");
  const n = size[0];
  const cofactor = [];
  for (let i = 0; i < n; i++) {
    const row = [];
    for (let j = 0; j < n; j++) {
      const minor = [];
      for (let r = 0; r < n; r++) {
        if (r === i) continue;
        const minorRow = [];
        for (let c = 0; c < n; c++) {
          if (c === j) continue;
          minorRow.push(m[r][c]);
        }
        minor.push(minorRow);
      }
      const minorDet = math.det(minor);
      const sign = ((i + j) % 2 === 0) ? 1 : -1;
      row.push(math.multiply(sign, minorDet));
    }
    cofactor.push(row);
  }
  return math.transpose(cofactor);
}

// preprocess expression: constants & deg/rad handling
function preprocessExpression(input, angleMode) {
  if (!input) return input;
  let s = String(input).trim();
  s = s.replace(/\bPI\b/gi, "pi").replace(/\bE\b/g, "e");
  s = s.replace(/(\d+(\.\d+)?)\s*°/g, "$1 deg");
  if (angleMode === "DEG") {
    const forward = ["sin", "cos", "tan", "sinh", "cosh", "tanh"];
    for (const fn of forward) {
      s = s.replace(new RegExp(fn + '\\s*\\(([^()]*)\\)', 'gi'), (m, g1) => {
        if (/\bdeg\b|unit\(/i.test(g1)) return `${fn}(${g1})`;
        return `${fn}(unit((${g1}),'deg'))`;
      });
    }
    const inverse = ["asin", "acos", "atan"];
    for (const fn of inverse) {
      s = s.replace(new RegExp(fn + '\\s*\\(([^()]*)\\)', 'gi'), (m, g1) => `(${fn}(${g1}) * 180 / pi)`);
    }
    s = s.replace(/(\b[\d.]+\b)\s*deg/gi, "unit($1, 'deg')");
  }
  return s;
}

// numeric integration (Simpson's rule) for definite integral
function numericIntegrate(f, a, b, n = 1000) {
  // ensure n even
  if (n % 2 === 1) n++;
  const h = (b - a) / n;
  let s = 0;
  for (let i = 0; i <= n; i++) {
    const x = a + i * h;
    const fx = f(x);
    if (i === 0 || i === n) s += fx;
    else if (i % 2 === 1) s += 4 * fx;
    else s += 2 * fx;
  }
  return (s * h) / 3;
}

// ----------------- React component -----------------
export default function ScientificCalculator() {
  // mode tabs
  const [mode, setMode] = useState("matrix"); // matrix | vector | calculus | scientific
  const [angleMode, setAngleMode] = useState("DEG");
  const [preferFraction, setPreferFraction] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    // Check if user has a saved preference
    const saved = localStorage.getItem('darkMode');
    if (saved !== null) {
      return JSON.parse(saved);
    }
    // Check system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  // common history
  const [history, setHistory] = useState([]);

  // ---- Scientific ----
  const [sciInput, setSciInput] = useState("");
  const [sciResult, setSciResult] = useState("");

  // ---- Matrix ----
  const [rows, setRows] = useState(2);
  const [cols, setCols] = useState(2);
  const gridRef = useRef([]);
  const [gridKey, setGridKey] = useState(0);
  const [matrixResult, setMatrixResult] = useState(null); // keep as array or primitive
  const [matrixResultString, setMatrixResultString] = useState("");
  const [otherRows, setOtherRows] = useState(2);
  const [otherCols, setOtherCols] = useState(2);
  const otherGridRef = useRef([]);
  const [otherVisible, setOtherVisible] = useState(false);
  const [selectedMatOp, setSelectedMatOp] = useState("det");
  const [powN, setPowN] = useState(2);

  // ---- Vector ----
  const [vecA, setVecA] = useState("1,0,0");
  const [vecB, setVecB] = useState("0,1,0");
  const [vecC, setVecC] = useState("0,0,1");
  const [vecResult, setVecResult] = useState("");
  const [planeMode, setPlaneMode] = useState("points"); // points | pointnormal
  const [planeP1, setPlaneP1] = useState("0,0,0");
  const [planeP2, setPlaneP2] = useState("1,0,0");
  const [planeP3, setPlaneP3] = useState("0,1,0");
  const [planePoint, setPlanePoint] = useState("0,0,0");
  const [planeNormal, setPlaneNormal] = useState("0,0,1");
  const [planeResult, setPlaneResult] = useState("");

  // ---- Calculus ----
  const [calcExpr, setCalcExpr] = useState("x^2");
  const [calcVar, setCalcVar] = useState("x");
  const [calcDiffResult, setCalcDiffResult] = useState("");
  const [calcPartialExpr, setCalcPartialExpr] = useState("x^2 + y^2");
  const [calcPartialVars, setCalcPartialVars] = useState("x,y");
  const [calcPartialResult, setCalcPartialResult] = useState("");
  const [calcIntA, setCalcIntA] = useState("0");
  const [calcIntB, setCalcIntB] = useState("1");
  const [calcIntN, setCalcIntN] = useState(1000);
  const [calcIntResult, setCalcIntResult] = useState("");

  // --- utility hooks to ensure refs grid shape ---
  useEffect(() => {
    gridRef.current = Array.from({ length: rows }, (_, r) =>
      Array.from({ length: cols }, (_, c) => gridRef.current?.[r]?.[c] || React.createRef())
    );
    setGridKey(k => k + 1);
  }, [rows, cols]);

  useEffect(() => {
    otherGridRef.current = Array.from({ length: otherRows }, (_, r) =>
      Array.from({ length: otherCols }, (_, c) => otherGridRef.current?.[r]?.[c] || React.createRef())
    );
  }, [otherRows, otherCols]);

  // Effect to save dark mode preference and apply theme
  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // ---------------- Scientific handlers ----------------
  function sciEvaluate() {
    if (!sciInput.trim()) return;
    try {
      const pre = preprocessExpression(sciInput, angleMode);
      const scope = { pi: Math.PI, e: Math.E };
      const v = mathSci.evaluate(pre, scope);
      const formatted = formatResult(v, preferFraction);
      setSciResult(formatted);
      setHistory(h => [{ mode: "sci", in: sciInput, out: formatted }, ...h].slice(0, 200));
    } catch (err) {
      setSciResult("Error: " + (err.message || String(err)));
    }
  }

  // ---------------- Matrix helpers ----------------
  function readGrid(refMatrix, rCount, cCount) {
    if (!rCount || !cCount) return null;
    const mat = Array.from({ length: rCount }, () => Array(cCount).fill(0));
    for (let r = 0; r < rCount; r++) {
      for (let c = 0; c < cCount; c++) {
        const refObj = refMatrix.current?.[r]?.[c];
        const raw = refObj?.current?.value?.trim?.() ?? "";
        if (!raw) { mat[r][c] = 0; continue; }
        try {
          const pre = preprocessExpression(raw, angleMode);
          const val = math.evaluate(pre);
          mat[r][c] = val;
        } catch (err) {
          throw new Error(`Cell [${r+1},${c+1}] parse error: ${err.message || err}`);
        }
      }
    }
    return mat;
  }

  function runMatrixOp() {
    try {
      const A = readGrid(gridRef, rows, cols);
      if (!A) { setMatrixResultString("Create matrix A"); return; }
      const op = selectedMatOp; let out;
      if (op === "det") {
        if (rows !== cols) throw new Error("det requires square matrix");
        out = math.det(A);
        setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return;
      }
      if (op === "inv") { if (rows !== cols) throw new Error("inv requires square matrix"); out = math.inv(A); setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return; }
      if (op === "transpose") { out = math.transpose(A); setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return; }
      if (op === "adj") { if (rows !== cols) throw new Error("adj requires square matrix"); out = adjointMatrix(A); setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return; }
      if (op === "rank") { out = math.rank(A); setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return; }
      if (op === "pow") { if (!Number.isInteger(Number(powN))) throw new Error("exponent must be integer"); if (rows !== cols) throw new Error("power requires square matrix"); out = math.pow(A, Number(powN)); setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return; }
      if (["add","sub","mul"].includes(op)) {
        const B = readGrid(otherGridRef, otherRows, otherCols);
        if (!B) throw new Error("Create other matrix B first");
        if (op === "add" || op === "sub") {
          if (rows !== otherRows || cols !== otherCols) throw new Error("add/sub requires same dimensions");
          out = op === "add" ? math.add(A, B) : math.subtract(A, B);
          setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return;
        }
        if (op === "mul") {
          if (cols !== otherRows) throw new Error("A.cols must equal B.rows for multiplication");
          out = math.multiply(A, B);
          setMatrixResult(out); setMatrixResultString(formatResult(out, preferFraction)); return;
        }
      }
      throw new Error("unknown op");
    } catch (err) {
      setMatrixResultString("Error: " + (err.message || String(err)));
    }
  }

  function exportMatrixResultCSV() {
    try {
      if (!Array.isArray(matrixResult)) return alert("Result is not a matrix");
      const csv = matrixResult.map(row => row.map(cell => {
        try { return math.format(cell, { fraction: "ratio", precision: 12 }).replace(/,/g, ""); } catch { return String(cell); }
      }).join(",")).join("\n");
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a"); a.href = url; a.download = "matrix-result.csv"; document.body.appendChild(a);
      a.click(); a.remove(); URL.revokeObjectURL(url);
    } catch (e) { alert("Export failed: " + (e.message || e)); }
  }

  // ---------------- Vector features ----------------
  function parseVec(s) {
    // accepts "1,2,3" or "[1,2,3]" with expressions
    if (!s) return [];
    let trimmed = s.trim();
    if (trimmed.startsWith("[")) trimmed = trimmed.slice(1);
    if (trimmed.endsWith("]")) trimmed = trimmed.slice(0, -1);
    const parts = trimmed.split(",").map(p => p.trim()).filter(Boolean);
    return parts.map(p => {
      const pre = preprocessExpression(p, angleMode);
      return math.evaluate(pre);
    });
  }

  function vecToString(v) {
    if (!Array.isArray(v)) return String(v);
    return "[" + v.map(c => {
      try { return math.format(c, { fraction: "ratio", precision: 12 }); } catch { return String(c); }
    }).join(", ") + "]";
  }

  function runVectorOp(op) {
    try {
      const A = parseVec(vecA), B = parseVec(vecB), C = parseVec(vecC);
      if (op === "dot") {
        if (A.length !== B.length) throw new Error("dot: vectors must be same length");
        const s = math.dot(A, B);
        setVecResult(formatResult(s, preferFraction)); return;
      }
      if (op === "cross") {
        if (A.length !== 3 || B.length !== 3) throw new Error("cross: requires 3D vectors");
        const r = math.cross(A, B);
        setVecResult(vecToString(r)); return;
      }
      if (op === "scalarTriple") {
        if (A.length !== 3 || B.length !== 3 || C.length !== 3) throw new Error("scalar triple requires 3D vectors");
        const val = math.dot(A, math.cross(B, C));
        setVecResult(formatResult(val, preferFraction)); return;
      }
      if (op === "vectorTriple") {
        if (A.length !== 3 || B.length !== 3 || C.length !== 3) throw new Error("vector triple requires 3D vectors");
        const r = math.cross(A, math.cross(B, C)); // A x (B x C)
        setVecResult(vecToString(r)); return;
      }
      if (op === "resultant") {
        // sum of A,B,C depending on what provided
        const list = [A,B,C].filter(x => x && x.length);
        if (list.length === 0) throw new Error("No vectors provided");
        // ensure same length: pad zeros
        const maxL = Math.max(...list.map(l => l.length));
        const padded = list.map(v => Array.from({length: maxL}, (_,i)=> v[i] ?? 0));
        const sum = padded.reduce((acc,row)=> acc.map((x,i)=> math.add(x, row[i])), Array(maxL).fill(0));
        setVecResult(vecToString(sum)); return;
      }
      if (op === "magnitude") {
        const v = parseVec(vecA);
        const mag = Math.sqrt(v.reduce((s,x)=> s + Math.pow(Number(math.evaluate(math.format(x))),2), 0));
        setVecResult(formatDecimal(mag)); return;
      }
      if (op === "normalize") {
        const v = parseVec(vecA);
        const arr = v.map(x => Number(math.evaluate(math.format(x))));
        const mag = Math.sqrt(arr.reduce((s,x)=> s + x*x, 0));
        if (mag === 0) throw new Error("zero vector");
        const norm = arr.map(x => x / mag);
        setVecResult("[" + norm.map(x=>formatDecimal(x)).join(", ") + "]"); return;
      }
      if (op === "angleBetween") {
        if (A.length !== B.length) throw new Error("vectors must match length");
        const dot = Number(math.evaluate(math.format(math.dot(A,B))));
        const magA = Math.sqrt(A.reduce((s,x)=> s + Math.pow(Number(math.evaluate(math.format(x))),2),0));
        const magB = Math.sqrt(B.reduce((s,x)=> s + Math.pow(Number(math.evaluate(math.format(x))),2),0));
        const cos = dot / (magA * magB);
        const angleRad = Math.acos(Math.max(-1, Math.min(1, cos)));
        if (angleMode === "DEG") setVecResult(formatDecimal(angleRad * 180 / Math.PI) + "°"); else setVecResult(formatDecimal(angleRad) + " rad");
        return;
      }
      if (op === "projection") {
        // projection of A onto B: (A·B / B·B) B
        if (A.length !== B.length) throw new Error("vectors must match length");
        const denom = Number(math.evaluate(math.format(math.dot(B,B))));
        if (denom === 0) throw new Error("projection onto zero vector");
        const factor = Number(math.evaluate(math.format(math.dot(A,B)))) / denom;
        const proj = B.map(x => math.multiply(x, factor));
        setVecResult(vecToString(proj)); return;
      }
      throw new Error("unknown vector op");
    } catch (err) {
      setVecResult("Error: " + (err.message || String(err)));
    }
  }

  // ---- Plane features ----
  function planeFromThreePoints() {
    try {
      const p1 = parseVec(planeP1), p2 = parseVec(planeP2), p3 = parseVec(planeP3);
      if (p1.length !== 3 || p2.length !== 3 || p3.length !== 3) throw new Error("points must be 3D");
      const v1 = math.subtract(p2, p1), v2 = math.subtract(p3, p1);
      const normal = math.cross(v1, v2);
      const [A,B,C] = normal.map(x => Number(math.evaluate(math.format(x))));
      const D = -(A * Number(math.evaluate(math.format(p1[0]))) + B * Number(math.evaluate(math.format(p1[1]))) + C * Number(math.evaluate(math.format(p1[2]))));
      const eq = `${formatDecimal(A)}x + ${formatDecimal(B)}y + ${formatDecimal(C)}z + ${formatDecimal(D)} = 0`;
      setPlaneResult(`Normal: ${vecToString(normal)}\nEquation: ${eq}`);
    } catch (err) {
      setPlaneResult("Error: " + (err.message || String(err)));
    }
  }
  function planeFromPointNormal() {
    try {
      const p = parseVec(planePoint), n = parseVec(planeNormal);
      if (p.length !== 3 || n.length !== 3) throw new Error("point and normal must be 3D");
      const [A,B,C] = n.map(x => Number(math.evaluate(math.format(x))));
      const D = -(A * Number(math.evaluate(math.format(p[0]))) + B * Number(math.evaluate(math.format(p[1]))) + C * Number(math.evaluate(math.format(p[2]))));
      const eq = `${formatDecimal(A)}x + ${formatDecimal(B)}y + ${formatDecimal(C)}z + ${formatDecimal(D)} = 0`;
      setPlaneResult(`Normal: ${vecToString(n)}\nEquation: ${eq}`);
    } catch (err) {
      setPlaneResult("Error: " + (err.message || String(err)));
    }
  }
  function pointOnPlaneCheck(ptStr) {
    try {
      const p = parseVec(ptStr);
      if (p.length !== 3) throw new Error("point must be 3D");
      // prefer plane from three points if that was used; check which mode
      if (planeMode === "points") {
        const p1 = parseVec(planeP1), p2 = parseVec(planeP2), p3 = parseVec(planeP3);
        const v1 = math.subtract(p2, p1), v2 = math.subtract(p3, p1);
        const normal = math.cross(v1, v2);
        const val = Number(math.evaluate(math.format(math.dot(normal, math.subtract(p, p1)))));
        setPlaneResult(`Point on plane? ${Math.abs(val) < 1e-9 ? "Yes" : "No"} (dot(normal, p - p1) = ${formatDecimal(val)})`);
      } else {
        const n = parseVec(planeNormal); const P = parseVec(planePoint);
        const val = Number(math.evaluate(math.format(math.dot(n, math.subtract(p, P)))));
        setPlaneResult(`Point on plane? ${Math.abs(val) < 1e-9 ? "Yes" : "No"} (dot(normal, p - P) = ${formatDecimal(val)})`);
      }
    } catch (err) { setPlaneResult("Error: " + (err.message || String(err))); }
  }

  // ---------------- Calculus ----------------
  function differentiate() {
    try {
      const pre = preprocessExpression(calcExpr, angleMode);
      // mathjs derivative
      const d = math.derivative(pre, calcVar);
      const simplified = d.toString();
      setCalcDiffResult(simplified);
      setHistory(h => [{ mode: "diff", in: `${calcExpr} d/d${calcVar}`, out: simplified }, ...h].slice(0,200));
    } catch (err) { setCalcDiffResult("Error: " + (err.message || String(err))); }
  }

  function partialDifferentiate() {
    try {
      const vars = calcPartialVars.split(",").map(s => s.trim()).filter(Boolean);
      if (vars.length === 0) throw new Error("enter partial variable names comma separated");
      let expr = calcPartialExpr;
      let out = vars.map(v => `${v}: ${math.derivative(expr, v).toString()}`).join("\n");
      setCalcPartialResult(out);
      setHistory(h => [{ mode: "pdiff", in: `${expr} d/d${vars.join(",")}`, out }, ...h].slice(0,200));
    } catch (err) { setCalcPartialResult("Error: " + (err.message || String(err))); }
  }

  function definiteIntegrate() {
    try {
      const pre = preprocessExpression(calcExpr, angleMode);
      const f = (x) => {
        // evaluate expression with x -> numeric
        const scope = { [calcVar]: x, pi: Math.PI, e: Math.E };
        try {
          // math.evaluate with scope
          const res = math.evaluate(pre, scope);
          return Number(math.evaluate(math.format(res)));
        } catch {
          // fallback numeric eval (not ideal)
          return 0;
        }
      };
      const a = Number(math.evaluate(preprocessExpression(calcIntA, angleMode)));
      const b = Number(math.evaluate(preprocessExpression(calcIntB, angleMode)));
      if (!isFinite(a) || !isFinite(b)) throw new Error("invalid bounds");
      const n = Math.max(10, Math.min(20000, Number(calcIntN) || 1000));
      const val = numericIntegrate(f, a, b, n);
      setCalcIntResult(formatDecimal(val));
      setHistory(h => [{ mode: "int", in: `${calcExpr} from ${a} to ${b}`, out: formatDecimal(val) }, ...h].slice(0,200));
    } catch (err) { setCalcIntResult("Error: " + (err.message || String(err))); }
  }

  // ---------------- Render helpers ----------------
  function renderMatrixAsTable(mat, preferFraction = true) {
    if (!Array.isArray(mat)) return <div className="text-sm">{String(mat)}</div>;
    return (
      <table className="border-collapse border">
        <tbody>
          {mat.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j} className="px-2 py-1 border text-sm mono">
                  {(() => {
                    try { return math.format(cell, { fraction: (preferFraction ? 'ratio' : false), precision: 12 }); }
                    catch { return String(cell); }
                  })()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  // ---------------- UI ----------------
  return (
    <div className={`min-h-screen p-6 transition-colors duration-200 ${darkMode ? 'bg-gray-900' : 'bg-slate-50'}`} 
         style={{
           backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100' fill='none'%3E%3Ccircle cx='20' cy='20' r='1' fill='%23cbd5e1' fill-opacity='0.1'/%3E%3Ccircle cx='80' cy='40' r='1' fill='%23cbd5e1' fill-opacity='0.1'/%3E%3Ccircle cx='40' cy='80' r='1' fill='%23cbd5e1' fill-opacity='0.1'/%3E%3Ccircle cx='90' cy='90' r='1' fill='%23cbd5e1' fill-opacity='0.1'/%3E%3C/svg%3E")`,
           backgroundSize: '100px 100px'
         }}>
      <div className="max-w-6xl mx-auto">
        <div className={`rounded-2xl shadow p-6 space-y-4 transition-colors duration-200 ${darkMode ? 'bg-gray-800 text-white' : 'bg-white'} relative z-10`}>
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-semibold">Calculator — Matrix • Vector • Calculus • Scientific</h1>
            <div className="flex gap-2 items-center">
              <div className="text-sm">Mode:</div>
              <div className={`rounded px-2 py-1 ${darkMode ? 'bg-gray-700' : 'bg-amber-50'}`}>
                <button className={`px-2 py-1 ${mode==='matrix' ? 'bg-indigo-600 text-white rounded' : ''}`} onClick={() => setMode('matrix')}>Matrix</button>
                <button className={`px-2 py-1 ${mode==='vector' ? 'bg-indigo-600 text-white rounded' : ''}`} onClick={() => setMode('vector')}>Vector</button>
                <button className={`px-2 py-1 ${mode==='calculus' ? 'bg-indigo-600 text-white rounded' : ''}`} onClick={() => setMode('calculus')}>Calculus</button>
                <button className={`px-2 py-1 ${mode==='scientific' ? 'bg-indigo-600 text-white rounded' : ''}`} onClick={() => setMode('scientific')}>Scientific</button>
              </div>
            </div>
          </div>

          <div className="flex gap-6 items-center">
            <div>
              <div className="text-sm">Angle</div>
              <button className={`mt-1 px-3 py-1 rounded transition-colors duration-200 ${darkMode ? 'bg-gray-700 hover:bg-gray-600' : 'bg-slate-100 hover:bg-slate-200'}`} onClick={() => setAngleMode(a => a==='DEG' ? 'RAD' : 'DEG')}>{angleMode}</button>
            </div>
            <div>
              <div className="text-sm">Prefer fractions</div>
              <label className="inline-flex items-center gap-2 mt-1">
                <input type="checkbox" checked={preferFraction} onChange={(e) => setPreferFraction(e.target.checked)} />
                <span className="text-sm">{preferFraction ? 'on' : 'off'}</span>
              </label>
            </div>
            <div className="ml-auto text-sm text-slate-500 dark:text-gray-400">Matrix result can be exported to CSV and shows a rendered visual grid.</div>
          </div>

          {/* Dark mode toggle */}
          <div className="flex justify-end">
            <button
              onClick={() => setDarkMode(!darkMode)}
              className={`px-4 py-2 rounded-lg transition-colors duration-200 flex items-center gap-2 ${
                darkMode
                  ? 'bg-yellow-500 hover:bg-yellow-400 text-gray-900'
                  : 'bg-gray-700 hover:bg-gray-600 text-white'
              }`}
            >
              {darkMode ? (
                <>
                  <span>☀️</span>
                  <span>Light Mode</span>
                </>
              ) : (
                <>
                  <span>🌙</span>
                  <span>Dark Mode</span>
                </>
              )}
            </button>
          </div>

          {/* ---------- SCIENTIFIC ---------- */}
          {mode === "scientific" && (
            <div className="space-y-3 relative">
              {/* Background Calculator Image */}
              <div className="absolute inset-0 bg-gradient-to-br from-blue-50/30 to-purple-50/30 dark:from-blue-900/20 dark:to-purple-900/20 rounded-2xl pointer-events-none" 
                   style={{
                     backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 200 200' fill='none'%3E%3Crect x='20' y='20' width='160' height='160' rx='10' fill='%23e2e8f0' fill-opacity='0.1'/%3E%3Crect x='30' y='40' width='140' height='30' rx='5' fill='%23cbd5e1' fill-opacity='0.2'/%3E%3Crect x='30' y='80' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='70' y='80' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='110' y='80' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='150' y='80' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='30' y='115' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='70' y='115' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='110' y='115' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='150' y='115' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='30' y='150' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='70' y='150' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='110' y='150' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3Crect x='150' y='150' width='30' height='25' rx='3' fill='%2394a3b8' fill-opacity='0.3'/%3E%3C/svg%3E")`,
                     backgroundSize: '200px 200px',
                     backgroundPosition: 'center',
                     backgroundRepeat: 'no-repeat',
                     opacity: '0.1'
                   }}>
              </div>
              
              {/* Instructions */}
              <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-blue-900 border-blue-700' : 'bg-blue-50 border-blue-200'} relative z-10`}>
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">💡 How to use the Scientific Calculator</h4>
                <div className="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                  <div>• Type expressions directly or use the buttons below</div>
                  <div>• Press Enter or click Evaluate to calculate</div>
                  <div>• Use parentheses for grouping: (2+3)*4</div>
                  <div>• Functions: sin(30), sqrt(16), factorial(5)</div>
                  <div>• Constants: pi, e</div>
                  <div>• Operations: +, -, *, /, ^ (power)</div>
                </div>
              </div>
              
              <div className="bg-slate-900 text-white p-3 rounded relative z-10">
                <input value={sciInput} onChange={e => setSciInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') sciEvaluate(); }} placeholder="e.g. sin(30) or (1/2 + sqrt(2))" className="w-full bg-transparent outline-none text-xl mono" />
                <div className="mt-2 text-right text-sm text-slate-300">
                  Result: <span className="font-semibold mono">{sciResult}</span>
                  <span className="ml-4 text-xs text-slate-400">Press Enter to evaluate</span>
                </div>
              </div>
              
              {/* Scientific Calculator Keypad */}
              <div className={`rounded-xl border ${darkMode ? 'border-gray-700 bg-gray-800' : 'border-slate-200 bg-white'} p-4 mx-auto max-w-4xl relative z-10`}> 
                <div className={`sci-pad grid gap-3 ${darkMode ? 'text-white' : ''} grid-cols-4 sm:grid-cols-6 lg:grid-cols-7 place-items-center`}>
                {/* Row 1: Scientific Functions */}
                <button className="function" onClick={() => setSciInput(s => s + 'sin(')}>sin</button>
                <button className="function" onClick={() => setSciInput(s => s + 'cos(')}>cos</button>
                <button className="function" onClick={() => setSciInput(s => s + 'tan(')}>tan</button>
                <button className="function" onClick={() => setSciInput(s => s + 'asin(')}>asin</button>
                <button className="function" onClick={() => setSciInput(s => s + 'acos(')}>acos</button>
                <button className="function" onClick={() => setSciInput(s => s + 'atan(')}>atan</button>
                <button className="function" onClick={() => setSciInput(s => s + 'log(')}>log</button>
                
                {/* Row 2: More Scientific Functions */}
                <button className="function" onClick={() => setSciInput(s => s + 'ln(')}>ln</button>
                <button className="function" onClick={() => setSciInput(s => s + 'sqrt(')}>√</button>
                <button className="function" onClick={() => setSciInput(s => s + 'cbrt(')}>∛</button>
                <button className="function" onClick={() => setSciInput(s => s + 'abs(')}>|x|</button>
                <button className="function" onClick={() => setSciInput(s => s + 'floor(')}>⌊x⌋</button>
                <button className="function" onClick={() => setSciInput(s => s + 'ceil(')}>⌈x⌉</button>
                <button className="function" onClick={() => setSciInput(s => s + 'round(')}>round</button>
                
                {/* Row 3: Numbers and Basic Operations */}
                <button className="number" onClick={() => setSciInput(s => s + '7')}>7</button>
                <button className="number" onClick={() => setSciInput(s => s + '8')}>8</button>
                <button className="number" onClick={() => setSciInput(s => s + '9')}>9</button>
                <button className="operator" onClick={() => setSciInput(s => s + '/')}>/</button>
                <button className="operator" onClick={() => setSciInput(s => s + '^')}>^</button>
                <button className="operator" onClick={() => setSciInput(s => s + '(')}>(</button>
                <button className="operator" onClick={() => setSciInput(s => s + ')')}>)</button>
                
                {/* Row 4: More Numbers and Operations */}
                <button className="number" onClick={() => setSciInput(s => s + '4')}>4</button>
                <button className="number" onClick={() => setSciInput(s => s + '5')}>5</button>
                <button className="number" onClick={() => setSciInput(s => s + '6')}>6</button>
                <button className="operator" onClick={() => setSciInput(s => s + '*')}>×</button>
                <button className="special" onClick={() => setSciInput(s => s + '!')}>!</button>
                <button className="special" onClick={() => setSciInput(s => s + 'pi')}>π</button>
                <button className="special" onClick={() => setSciInput(s => s + 'e')}>e</button>
                
                {/* Row 5: Final Numbers and Operations */}
                <button className="number" onClick={() => setSciInput(s => s + '1')}>1</button>
                <button className="number" onClick={() => setSciInput(s => s + '2')}>2</button>
                <button className="number" onClick={() => setSciInput(s => s + '3')}>3</button>
                <button className="number" onClick={() => setSciInput(s => s + '0')}>0</button>
                <button className="number" onClick={() => setSciInput(s => s + '.')}>.</button>
                <button className="operator" onClick={() => setSciInput(s => s + '-')}>-</button>
                <button className="operator" onClick={() => setSciInput(s => s + '+')}>+</button>
                
                {/* Row 6: Control Buttons */}
                <button className="special" title="random 0..1" onClick={() => setSciInput(s => s + 'random()')}>rand</button>
                <button className="special" title="modulo" onClick={() => setSciInput(s => s + ' mod ')}>mod</button>
                <button className="special" title="gcd(a,b)" onClick={() => setSciInput(s => s + 'gcd(')}>gcd</button>
                <button className="special" title="lcm(a,b)" onClick={() => setSciInput(s => s + 'lcm(')}>lcm</button>
                <button className="special" title="degree symbol" onClick={() => setSciInput(s => s + '°')}>°</button>
                <button className="special" title="comma" onClick={() => setSciInput(s => s + ',') }>,</button>
                
                {/* Row 7: Special Functions and Clear */}
                <button className="function" onClick={() => setSciInput(s => s + 'exp(')}>exp</button>
                <button className="function" onClick={() => setSciInput(s => s + 'sinh(')}>sinh</button>
                <button className="function" onClick={() => setSciInput(s => s + 'cosh(')}>cosh</button>
                <button className="function" onClick={() => setSciInput(s => s + 'tanh(')}>tanh</button>
                <button className="function" onClick={() => setSciInput(s => s + 'log10(')}>log10</button>
                <button className="function" onClick={() => setSciInput(s => s + 'log2(')}>log2</button>
                <button className="clear" onClick={() => { setSciInput(''); setSciResult(''); }}>C</button>
                
                {/* Row 8: Special Functions */}
                <button className="function" onClick={() => setSciInput(s => s + 'factorial(')}>fact</button>
                <button className="function" onClick={() => setSciInput(s => s + 'combinations(')}>nCr</button>
                <button className="function" onClick={() => setSciInput(s => s + 'permutations(')}>nPr</button>
                <button className="function" onClick={() => setSciInput(s => s + 'binomial(')}>binom</button>
                <button className="function" onClick={() => setSciInput(s => s + 'fibonacci(')}>fib</button>
                <button className="function" onClick={() => setSciInput(s => s + 'prime(')}>prime</button>
              </div>
              </div>
              
              {/* Quick Examples and Evaluate */}
              <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                <div className="flex justify-between items-center mb-2">
                  <h4 className="font-medium">Quick Examples</h4>
                  <button 
                    className={`px-6 py-2 rounded-lg transition-colors duration-200 bg-indigo-600 hover:bg-indigo-700 text-white font-medium`} 
                    onClick={sciEvaluate}
                  >
                    Evaluate
                  </button>
                </div>
                <div className="example-pad grid grid-cols-3 gap-3 text-sm mx-auto max-w-4xl">
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('sin(30)')}>sin(30°)</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('2^10')}>2^10</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('sqrt(16)')}>√16</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('factorial(5)')}>5!</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('log(100)')}>log(100)</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('pi * 2^2')}>π×r²</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('1 + 2 * 3')}>1+2×3</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('e^2')}>e²</button>
                  <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={() => setSciInput('abs(-5)')}>|-5|</button>
                </div>
              </div>
            </div>
          )}

          {/* ---------- MATRIX ---------- */}
          {mode === "matrix" && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                  <h3 className="font-medium">Create Matrix A</h3>
                  <div className="flex gap-2 items-center mt-2">
                    <label className="text-sm">Rows</label>
                    <input type="number" min="1" max="8" value={rows} onChange={e=>setRows(Math.max(1,Math.min(8,Number(e.target.value||1))))} className={`w-16 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} />
                    <label className="text-sm">Cols</label>
                    <input type="number" min="1" max="8" value={cols} onChange={e=>setCols(Math.max(1,Math.min(8,Number(e.target.value||1))))} className={`w-16 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} />
                    <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>setGridKey(k=>k+1)}>Create</button>
                    <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={()=>{ gridRef.current?.forEach(row=>row?.forEach(ref=>{ if(ref?.current) ref.current.value=''; })); }}>Clear</button>
                  </div>

                  <div className="mt-3">
                    <label className="text-sm">Paste matrix literal</label>
                    <div className="flex gap-2 mt-2">
                      <input id="literal-input" className={`flex-1 px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} placeholder='[[1,2],[3,4]] or [[1/2, sqrt(2)], [pi,0]]' />
                      <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>{ const v = document.getElementById('literal-input').value; parseLiteralToGrid(v); }}>Parse</button>
                    </div>
                  </div>

                  <div className="mt-3">
                    <h4 className="font-medium">Save / Load</h4>
                    <SaveLoadUI gridRef={gridRef} rows={rows} cols={cols} setSavedList={setHistory} darkMode={darkMode} />
                  </div>
                </div>

                <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                  <h3 className="font-medium">Operations</h3>
                  <div className="mt-2">
                    <select className={`w-full px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={selectedMatOp} onChange={e=>{ setSelectedMatOp(e.target.value); setOtherVisible(['add','sub','mul'].includes(e.target.value)); }}>
                      <option value="det">Determinant</option>
                      <option value="inv">Inverse</option>
                      <option value="transpose">Transpose</option>
                      <option value="adj">Adjoint</option>
                      <option value="rank">Rank</option>
                      <option value="mul">Multiply (A * B)</option>
                      <option value="add">Add (A + B)</option>
                      <option value="sub">Subtract (A - B)</option>
                      <option value="pow">Power A^n</option>
                    </select>
                  </div>

                  {otherVisible && (
                    <div className="mt-3">
                      <div className="text-sm">Other matrix B</div>
                      <div className="flex gap-2 mt-2 items-center">
                        <input type="number" className={`w-16 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} min="1" max="8" value={otherRows} onChange={e=>setOtherRows(Math.max(1,Math.min(8,Number(e.target.value||1))))} />
                        <input type="number" className={`w-16 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} min="1" max="8" value={otherCols} onChange={e=>setOtherCols(Math.max(1,Math.min(8,Number(e.target.value||1))))} />
                        <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>setGridKey(k=>k+1)}>Create B</button>
                      </div>
                    </div>
                  )}

                  {selectedMatOp === 'pow' && (
                    <div className="mt-3">
                      <div className="text-sm">Exponent (integer)</div>
                      <input type="number" className={`mt-2 w-28 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={powN} onChange={e=>setPowN(Number(e.target.value||0))} />
                    </div>
                  )}

                  <div className="mt-4 flex gap-2">
                    <button className="px-3 py-2 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={runMatrixOp}>Run</button>
                    <button className={`px-3 py-2 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>{ copyTextFromMatrixResult(matrixResult); }}>Copy</button>
                    <button className={`px-3 py-2 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={exportMatrixResultCSV}>Export CSV</button>
                  </div>
                </div>
              </div>

              <div className="flex gap-6 flex-wrap">
                <div>
                  <h4 className="font-medium mb-2">Matrix A</h4>
                  <div className={`p-2 rounded border inline-block transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}>
                    <MatrixGrid refObj={gridRef} rows={rows} cols={cols} key={gridKey} angleMode={angleMode} darkMode={darkMode} />
                  </div>
                </div>

                {otherVisible && (
                  <div>
                    <h4 className="font-medium mb-2">Matrix B</h4>
                    <div className={`p-2 rounded border inline-block transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-300'}`}>
                      <MatrixGrid refObj={otherGridRef} rows={otherRows} cols={otherCols} key={`${gridKey}-other`} angleMode={angleMode} darkMode={darkMode} />
                    </div>
                  </div>
                )}

                <div className="flex-1">
                  <h4 className="font-medium mb-2">Result (visual)</h4>
                  <div className="bg-slate-900 text-white p-3 rounded min-h-[120px]">
                    <div className="mb-2 text-sm">Rendered result (matrix) — or value below</div>
                    <div>{matrixResult ? renderMatrixAsTable(matrixResult, preferFraction) : <div className="mono">{matrixResultString || "No result yet"}</div>}</div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* ---------- VECTOR ---------- */}
          {mode === "vector" && (
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                <h3 className="font-medium">Vectors</h3>
                <div className="mt-2 text-sm">Enter vectors as comma-separated expressions (fractions & math allowed):</div>
                <div className="mt-2">
                  <label className="text-xs">Vector A</label>
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={vecA} onChange={e=>setVecA(e.target.value)} />
                </div>
                <div className="mt-2">
                  <label className="text-xs">Vector B</label>
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={vecB} onChange={e=>setVecB(e.target.value)} />
                </div>
                <div className="mt-2">
                  <label className="text-xs">Vector C (optional)</label>
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={vecC} onChange={e=>setVecC(e.target.value)} />
                </div>

                <div className="mt-3 grid grid-cols-3 gap-2">
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('dot')}>Dot</button>
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('cross')}>Cross</button>
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('scalarTriple')}>Scalar Triple</button>
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('vectorTriple')}>Vector Triple</button>
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('resultant')}>Resultant</button>
                  <button className="px-2 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={()=>runVectorOp('projection')}>Projection A→B</button>
                  <button className={`px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>runVectorOp('magnitude')}>|A|</button>
                  <button className={`px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>runVectorOp('normalize')}>Normalize A</button>
                  <button className={`px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>runVectorOp('angleBetween')}>Angle A-B</button>
                </div>

                <div className="mt-3">
                  <h4 className="font-medium">Planes</h4>
                  <div className="mt-2">
                    <label className="text-xs mr-2">Mode</label>
                    <select value={planeMode} onChange={e=>setPlaneMode(e.target.value)} className={`px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`}>
                      <option value="points">Three points</option>
                      <option value="pointnormal">Point + Normal</option>
                    </select>
                  </div>

                  {planeMode === 'points' ? (
                    <div className="mt-2 space-y-2">
                      <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={planeP1} onChange={e=>setPlaneP1(e.target.value)} placeholder="p1 e.g. 0,0,0" />
                      <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={planeP2} onChange={e=>setPlaneP2(e.target.value)} placeholder="p2 e.g. 1,0,0" />
                      <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={planeP3} onChange={e=>setPlaneP3(e.target.value)} placeholder="p3 e.g. 0,1,0" />
                      <div className="flex gap-2">
                        <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={planeFromThreePoints}>Make plane</button>
                        <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>pointOnPlaneCheck(prompt('Point to check (x,y,z)'))}>Check point</button>
                      </div>
                    </div>
                  ) : (
                    <div className="mt-2 space-y-2">
                      <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={planePoint} onChange={e=>setPlanePoint(e.target.value)} placeholder="point e.g. 0,0,0" />
                      <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={planeNormal} onChange={e=>setPlaneNormal(e.target.value)} placeholder="normal e.g. 0,0,1" />
                      <div className="flex gap-2">
                        <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={planeFromPointNormal}>Make plane</button>
                        <button className={`px-3 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-slate-100 hover:bg-slate-200 border-gray-300'}`} onClick={()=>pointOnPlaneCheck(prompt('Point to check (x,y,z)'))}>Check point</button>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              <div>
                <h4 className="font-medium">Result</h4>
                <div className="bg-slate-900 text-white p-3 rounded mono">
                  {vecResult || planeResult || "No result yet"}
                </div>
              </div>
            </div>
          )}

          {/* ---------- CALCULUS ---------- */}
          {mode === "calculus" && (
            <div className="grid grid-cols-2 gap-4">
              <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                <h3 className="font-medium">Differentiation</h3>
                <div className="mt-2">
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcExpr} onChange={e=>setCalcExpr(e.target.value)} placeholder="expression in x" />
                  <div className="mt-2 flex gap-2 items-center">
                    <label className="text-sm">Variable</label>
                    <input className={`w-24 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcVar} onChange={e=>setCalcVar(e.target.value)} />
                    <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={differentiate}>Differentiate</button>
                  </div>
                  <div className={`mt-2 p-2 rounded transition-colors duration-200 ${darkMode ? 'bg-gray-600' : 'bg-white'}`}>{calcDiffResult || "Result will appear here"}</div>
                </div>

                <h3 className="font-medium mt-4">Partial Differentiation</h3>
                <div className="mt-2">
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcPartialExpr} onChange={e=>setCalcPartialExpr(e.target.value)} placeholder="f(x,y,...) = ..." />
                  <div className="mt-2 flex gap-2 items-center">
                    <label className="text-sm">Variables (comma)</label>
                    <input className={`flex-1 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcPartialVars} onChange={e=>setCalcPartialVars(e.target.value)} />
                    <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={partialDifferentiate}>Partial Diff</button>
                  </div>
                  <div className={`mt-2 p-2 rounded transition-colors duration-200 ${darkMode ? 'bg-gray-600' : 'bg-white'}`}>{calcPartialResult || "Result will appear here"}</div>
                </div>
              </div>

              <div className={`p-3 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-slate-50 border-gray-300'}`}>
                <h3 className="font-medium">Integration (numeric)</h3>
                <div className="mt-2">
                  <input className={`w-full px-2 py-1 rounded border mono transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcExpr} onChange={e=>setCalcExpr(e.target.value)} placeholder="f(x)" />
                  <div className="mt-2 flex gap-2 items-center">
                    <label className="text-sm">a</label>
                    <input className={`w-24 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcIntA} onChange={e=>setCalcIntA(e.target.value)} />
                    <label className="text-sm">b</label>
                    <input className={`w-24 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcIntB} onChange={e=>setCalcIntB(e.target.value)} />
                    <label className="text-sm">n (steps)</label>
                    <input className={`w-20 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} value={calcIntN} onChange={e=>setCalcIntN(Number(e.target.value||100))} />
                  </div>
                  <div className="mt-2 flex gap-2">
                    <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={definiteIntegrate}>Integrate (numeric)</button>
                  </div>
                  <div className={`mt-2 p-2 rounded transition-colors duration-200 ${darkMode ? 'bg-gray-600' : 'bg-white'}`}>{calcIntResult || "Result will appear here"}</div>
                </div>
              </div>
            </div>
          )}

          {/* ---------- HISTORY ---------- */}
          <div>
            <h4 className="font-medium">History (recent)</h4>
            <div className={`max-h-32 overflow-auto mt-2 p-2 rounded transition-colors duration-200 ${darkMode ? 'bg-gray-700' : 'bg-slate-50'}`}>
              {history.length === 0 ? <div className="text-slate-500 dark:text-gray-400">No history yet</div> :
                history.slice(0,20).map((h, i) => (
                  <div key={i} className="text-sm mb-1">
                    <div className="text-xs text-slate-500 dark:text-gray-400">{h.mode}</div>
                    <div className="font-mono">{h.in}</div>
                    <div className="mono text-xs">{h.out}</div>
                  </div>
                ))
              }
            </div>
          </div>

        </div>
      </div>
    </div>
  );

  // ---------------- Inner small components & functions ----------------

  function MatrixGrid({ refObj, rows, cols, darkMode }) {
    // builds table of inputs connecting to refObj.current[r][c]
    const rowsArr = [];
    for (let r = 0; r < rows; r++) {
      const cells = [];
      for (let c = 0; c < cols; c++) {
        if (!refObj.current) refObj.current = [];
        if (!refObj.current[r]) refObj.current[r] = [];
        if (!refObj.current[r][c]) refObj.current[r][c] = React.createRef();
        cells.push(
          <td key={c} className="p-1 border">
            <input ref={refObj.current[r][c]} data-r={r} data-c={c} className={`matrix-cell w-28 px-2 py-1 rounded border mono text-sm transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`}
              onKeyDown={(e)=>matrixNavKey(e, refObj, r, c, rows, cols)} placeholder="0" />
          </td>
        );
      }
      rowsArr.push(<tr key={r}><th className="px-2 font-medium">R{r+1}</th>{cells}</tr>);
    }
    const headCols = [<th key="corner" className="px-2" />];
    for (let c = 0; c < cols; c++) headCols.push(<th key={`hc${c}`} className="px-2 font-medium">C{c+1}</th>);
    return (
      <table className="border-collapse">
        <thead><tr>{headCols}</tr></thead>
        <tbody>{rowsArr}</tbody>
      </table>
    );
  }

  function matrixNavKey(e, refMatrix, r, c, rCount, cCount) {
    const key = e.key;
    if (["ArrowUp","ArrowDown","ArrowLeft","ArrowRight","Enter","Tab"].includes(key)) {
      e.preventDefault();
      let nr = r, nc = c;
      if (key === "ArrowUp") nr = Math.max(0, r-1);
      if (key === "ArrowDown" || key === "Enter") nr = Math.min(rCount-1, r+1);
      if (key === "ArrowLeft") nc = Math.max(0, c-1);
      if (key === "ArrowRight" || key === "Tab") nc = Math.min(cCount-1, c+1);
      const nextRef = refMatrix.current?.[nr]?.[nc];
      if (nextRef && nextRef.current) nextRef.current.focus();
    }
  }



  function parseLiteralToGrid(lit) {
    if (!lit) return alert("enter literal");
    try {
      const evaluated = math.evaluate(lit);
      if (!Array.isArray(evaluated)) return alert("literal must be nested array");
      const rr = evaluated.length, cc = evaluated[0].length || 0;
      setRows(rr); setCols(cc);
      setTimeout(()=> {
        for (let r=0;r<rr;r++) for (let c=0;c<cc;c++) {
          const ref = gridRef.current?.[r]?.[c];
          if (ref?.current) {
            try { ref.current.value = math.format(evaluated[r][c], { fraction: 'ratio', precision: 12 }); } catch { ref.current.value = String(evaluated[r][c]); }
          }
        }
      }, 60);
    } catch (err) { alert("parse error: " + (err.message || String(err))); }
  }

  function copyTextFromMatrixResult(mat) {
    if (Array.isArray(mat)) {
      const txt = mat.map(r => r.map(cell => {
        try { return math.format(cell, { fraction: 'ratio', precision: 12 }).replace(/,/g,''); } catch { return String(cell); }
      }).join(",")).join("\n");
      navigator.clipboard?.writeText(txt);
      alert("copied");
    } else {
      navigator.clipboard?.writeText(String(mat));
      alert("copied");
    }
  }



  // Small save/load UI component
  function SaveLoadUI({ gridRef, rows, cols, setSavedList, darkMode }) {
    const [name, setName] = useState("");
    const [saved, setSaved] = useState(() => JSON.parse(localStorage.getItem("savedMatrices") || "[]"));

    function save() {
      if (!name) return alert("enter name");
      try {
        const A = readGrid(gridRef, rows, cols);
        if (!A) return alert("no matrix");
        const normalized = A.map(row => row.map(cell => math.format(cell, { fraction: 'ratio', precision: 12 })));
        const entry = { name, rows, cols, data: normalized, savedAt: Date.now() };
        const list = JSON.parse(localStorage.getItem("savedMatrices") || "[]");
        const idx = list.findIndex(x=>x.name===name);
        if (idx>=0) list[idx]=entry; else list.unshift(entry);
        localStorage.setItem("savedMatrices", JSON.stringify(list));
        setSaved(list); setSavedList(list);
        alert("saved");
      } catch (err) { alert("save failed: " + (err.message || String(err))); }
    }
    function load(n) {
      const list = JSON.parse(localStorage.getItem("savedMatrices") || "[]");
      const e = list.find(x=>x.name===n); if (!e) return alert("not found");
      // set rows/cols by prop callbacks — can't from here, so approximate: use window event
      // We'll set global rows/cols by dispatching event (simpler: call parseLiteralToGrid)
      // Build literal from saved data
      const literal = JSON.stringify(e.data);
      parseLiteralToGrid(literal);
    }
    function remove(n) {
      const list = JSON.parse(localStorage.getItem("savedMatrices") || "[]");
      const rem = list.filter(x=>x.name!==n);
      localStorage.setItem("savedMatrices", JSON.stringify(rem)); setSaved(rem); setSavedList(rem);
    }
    return (
      <div className="mt-2">
        <div className="flex gap-2">
          <input value={name} onChange={e=>setName(e.target.value)} className={`flex-1 px-2 py-1 rounded border transition-colors duration-200 ${darkMode ? 'bg-gray-600 border-gray-500 text-white' : 'bg-white border-gray-300'}`} placeholder="name" />
          <button className="px-3 py-1 bg-indigo-600 text-white rounded hover:bg-indigo-700 transition-colors duration-200" onClick={save}>Save</button>
        </div>
        <div className="mt-2 text-sm">
          {saved.length===0 ? <div className="text-slate-500 dark:text-gray-400">No saved matrices</div> : (
            <div className="space-y-1 max-h-44 overflow-auto">
              {saved.map(s=>(
                <div key={s.name} className="flex items-center gap-2">
                  <button className="text-left flex-1 text-sm" onClick={()=>load(s.name)}>{s.name} <span className="text-xs text-slate-400 dark:text-gray-500">({s.rows}x{s.cols})</span></button>
                  <button className={`px-2 py-1 rounded border text-xs transition-colors duration-200 ${darkMode ? 'bg-gray-600 hover:bg-gray-500 border-gray-500' : 'bg-white hover:bg-gray-50 border-gray-300'}`} onClick={()=>remove(s.name)}>Del</button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }

}
