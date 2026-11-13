// Combined UGP deck, v3 (Part I: RealPDE airfoil wake; Part II: Aurora in JAX).
// Action titles, one exhibit per slide, illustrative plots drawn at slide size by make_deck_figs.py and
// make_physics_figs.py. Usage: node build.js [out.pptx]   (also writes ../UGP_Final_speaker_notes.md)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const path = require("path");
const IMG = (f) => path.join(__dirname, "..", "figs", f);
const OUT = process.argv[2] || path.join(__dirname, "..", "UGP_Final_Presentation.pptx");

const INK = "2C3E50", BLUE = "2E75B6", OR = "CA6F1E", MUT = "5D6D7E", GREY = "AAB4BF", WHITE = "FFFFFF",
  PALE = "EAF1FB", PALEO = "FBEFE4", SOFT = "F2F5FA", LIGHT = "E2E8F0";
const FONT = "Arial";
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.author = "Aryamann Srivastava";
pres.title = "Teaching Neural Networks to Forecast Fluid Flow";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
const NOTES = require("./notes.json");
const TITLES = [];
let n = 0;

function dims(f) { // PNG width and height from the IHDR chunk
  const b = fs.readFileSync(IMG(f));
  return [b.readUInt32BE(16), b.readUInt32BE(20)];
}
function img(s, f, x, y, w, h, align) { // fit inside w x h keeping the aspect ratio; centred unless align = "left"
  const [pw, ph] = dims(f), r = pw / ph;
  let ww = w, hh = w / r;
  if (h && hh > h) { hh = h; ww = h * r; }
  s.addImage({ path: IMG(f), x: align === "left" ? x : x + (w - ww) / 2, y, w: ww, h: hh });
}
function slide(name, o) {
  o = o || {};
  const s = pres.addSlide();
  s.background = { color: o.dark ? INK : WHITE };
  n++;
  TITLES.push(name);
  s.addNotes(NOTES[n - 1]);
  if (!o.dark) {
    s.addText(String(n), { x: 9.1, y: 5.25, w: 0.4, h: 0.25, fontSize: 11, color: MUT, align: "right", fontFace: FONT, margin: 0, isTextBox: true });
    if (o.tag) s.addText(o.tag, { x: 0.5, y: 5.25, w: 6, h: 0.25, fontSize: 11, color: MUT, fontFace: FONT, margin: 0, isTextBox: true });
    if (o.cite) s.addText(o.cite, { x: 0.5, y: 4.97, w: 9, h: 0.25, fontSize: 11, color: MUT, fontFace: FONT, margin: 0, italic: true, isTextBox: true });
    s.addText(name, { x: 0.5, y: 0.28, w: 9, h: 0.95, fontSize: 26, bold: true, color: INK, fontFace: FONT, margin: 0, valign: "top", isTextBox: true });
  }
  return s;
}
const txt = (s, t, o) => s.addText(t, Object.assign({ fontFace: FONT, color: INK, fontSize: 18, margin: 0, valign: "top", isTextBox: true }, o));
function bullets(s, items, o) {
  const runs = [];
  items.forEach((it, i) => {
    const last = i === items.length - 1;
    if (Array.isArray(it)) runs.push({ text: it[0] + " ", options: { bold: true, bullet: { indent: 16 } } }, { text: it[1], options: { breakLine: !last } });
    else runs.push({ text: it, options: { bullet: { indent: 16 }, breakLine: !last } });
  });
  s.addText(runs, Object.assign({ fontFace: FONT, color: INK, fontSize: 18, margin: 0, valign: "top", paraSpaceAfter: 10, isTextBox: true }, o));
}
const fillBox = (s, x, y, w, h, fill) => s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill }, rectRadius: 0.06 });
const arrow = (s, x, y, w) => s.addShape(pres.shapes.RIGHT_ARROW, { x, y, w: w || 0.4, h: 0.28, fill: { color: GREY }, line: { color: GREY } });
function stat(s, x, y, w, big, label, color, size) {
  txt(s, big, { x, y, w, h: 0.75, fontSize: size || 40, bold: true, color: color || BLUE, align: "center", valign: "middle" });
  txt(s, label, { x, y: y + 0.8, w, h: 0.9, fontSize: 15, color: MUT, align: "center" });
}
function mathBox(s, t, y, o) {
  fillBox(s, 0.5, y, 9, 0.48, PALE);
  txt(s, t, Object.assign({ x: 0.5, y, w: 9, h: 0.48, fontSize: 16, bold: true, color: BLUE, align: "center", valign: "middle" }, o));
}
function divider(part, t, sub) {
  const s = slide(`${part}: ${t}`, { dark: true });
  txt(s, part, { x: 0.7, y: 1.6, w: 8.6, h: 0.5, fontSize: 18, bold: true, color: "9FC0EC" });
  txt(s, t, { x: 0.7, y: 2.1, w: 8.6, h: 0.9, fontSize: 34, bold: true, color: WHITE });
  txt(s, sub, { x: 0.7, y: 3.1, w: 8.6, h: 0.6, fontSize: 16, color: "C9D6E6" });
}
const P1 = "Part I  |  Real airfoil wake", P2 = "Part II  |  What pretraining data buys";

// 1 Title ------------------------------------------------------------------------------------------------------
{
  const s = slide("Teaching Neural Networks to Forecast Fluid Flow", { dark: true });
  txt(s, "Undergraduate Project  |  Parts I and II", { x: 0.6, y: 1.0, w: 5.6, h: 0.4, fontSize: 15, bold: true, color: "9FC0EC" });
  txt(s, "Teaching Neural Networks to Forecast Fluid Flow", { x: 0.6, y: 1.5, w: 5.6, h: 1.7, fontSize: 34, bold: true, color: WHITE });
  txt(s, "From a real airfoil wake to a rebuilt weather foundation model", { x: 0.6, y: 3.25, w: 5.4, h: 0.7, fontSize: 16, color: "C9D6E6" });
  txt(s, "Aryamann Srivastava  |  Aerospace Engineering, IIT Kanpur", { x: 0.6, y: 4.35, w: 5.6, h: 0.3, fontSize: 13, color: "C9D6E6" });
  txt(s, "Supervisor: Prof. Rajesh Ranjan  |  November 2026", { x: 0.6, y: 4.7, w: 5.6, h: 0.3, fontSize: 13, color: "C9D6E6" });
  s.addImage({ path: IMG("kolmo1.png"), x: 6.55, y: 0.65, w: 2.95, h: 2.95 });
  s.addImage({ path: IMG("wake.png"), x: 6.55, y: 3.75, w: 2.95, h: 2.95 / 2.168 });
}

// 2 Overview + the forecaster ----------------------------------------------------------------------------------
{
  const s = slide("A neural network can forecast a flow in a blink; we asked how to build one well", { tag: "Overview" });
  s.addImage({ path: IMG("wake.png"), x: 0.5, y: 1.35, w: 5.5, h: 5.5 / 2.168 });
  s.addImage({ path: IMG("kolmo1.png"), x: 6.95, y: 1.35, w: 2.54, h: 2.54 });
  txt(s, "Part I: a real airfoil wake, measured", { x: 0.5, y: 3.95, w: 5.5, h: 0.3, fontSize: 16, bold: true, color: BLUE });
  txt(s, "Part II: a flow we simulate", { x: 6.6, y: 3.95, w: 2.9, h: 0.3, fontSize: 16, bold: true, color: OR, align: "right" });
  mathBox(s, "next flow  ≈  f( flow now, flow just before )        repeat it to forecast many steps", 4.42);
}

// 3 Divider ----------------------------------------------------------------------------------------------------
divider("Part I", "Forecasting a Real Airfoil Wake", "NeurIPS 2026 RealPDE competition, Track 1: from simulation to reality");

// 4 The data ---------------------------------------------------------------------------------------------------
{
  const s = slide("Forecasting a real wake is hard: noisy data, unseen conditions, strict rules", { tag: P1, cite: "Data: RealPDE competition (Hu Z. et al., 2026). Blue: slow wake; red: fast flow." });
  img(s, "ugp1_data_c.png", 0.6, 1.2, 8.8, 3.3);
  txt(s, "81 recordings   |   256 MB limit   |   1 submission a day   |   unseen conditions", { x: 0.5, y: 4.55, w: 9, h: 0.3, fontSize: 15, bold: true, align: "center" });
}

// 5 Scoring weights --------------------------------------------------------------------------------------------
{
  const s = slide("Uncertainty was worth a quarter of the score; we recovered the hidden weights", { tag: P1 });
  s.addImage({ path: IMG("d_weights.png"), x: 0.5, y: 1.4, w: 5.6, h: 3.4 });
  bullets(s, ["Weights were never published", "Solved from 40 leaderboard scores", "Fit within 0.005 of every total"], { x: 6.5, y: 1.8, w: 3.0, h: 2.6, fontSize: 20 });
}

// 6 Uncertainty rule -------------------------------------------------------------------------------------------
{
  const s = slide("Only narrow bands that still contain the truth earn uncertainty points", { tag: P1 });
  const rows = [["Narrow, and correct", -0.005, 0.005, "0.84", BLUE], ["Medium, and correct", -0.015, 0.015, "0.59", BLUE], ["Wide, and correct", -0.05, 0.05, "0.17", GREY], ["Narrow, but misses", 0.05, 0.06, "0.00", OR]];
  const cx = 5.5, k = 24; // inches per unit of velocity (0.05 = 1.2 in)
  txt(s, "true value", { x: cx - 0.8, y: 1.3, w: 1.6, h: 0.3, fontSize: 14, color: MUT, align: "center" });
  rows.forEach(([lab, lo, hi, score, col], i) => {
    const y = 1.7 + i * 0.7;
    txt(s, lab, { x: 0.5, y, w: 2.9, h: 0.45, fontSize: 18, valign: "middle" });
    s.addShape(pres.shapes.RECTANGLE, { x: cx + lo * k, y: y + 0.04, w: Math.max((hi - lo) * k, 0.12), h: 0.37, fill: { color: col }, line: { color: col } });
    txt(s, score, { x: 8.3, y, w: 1.2, h: 0.45, fontSize: 22, bold: true, color: col === GREY ? MUT : col, valign: "middle", align: "right" });
  });
  s.addShape(pres.shapes.LINE, { x: cx, y: 1.6, w: 0, h: 2.85, line: { color: INK, width: 2 } });
  mathBox(s, "score = e^(−width / σ)  if the truth is inside the band, otherwise 0      (σ = 0.056)", 4.45);
}

// 7 The system ------------------------------------------------------------------------------------------------
{
  const s = slide("Two paths: a fine-tuned forecast, and uncertainty bands that never touch it", { tag: P1, cite: "FNO: Li et al., 2021  |  model soup: Wortsman et al., 2022  |  U-Net: Ronneberger et al., 2015" });
  const row = (y, label, boxes, col) => {
    txt(s, label, { x: 0.5, y: y - 0.36, w: 4, h: 0.3, fontSize: 15, bold: true, color: col });
    boxes.forEach(([h, d], i) => {
      const x = 0.5 + i * 2.35, first = i === 0, last = i === 3;
      fillBox(s, x, y, 1.95, 0.95, last ? col : first ? SOFT : PALE);
      txt(s, h, { x: x + 0.08, y: y + 0.08, w: 1.79, h: 0.35, fontSize: 14, bold: true, align: "center", color: last ? WHITE : INK });
      txt(s, d, { x: x + 0.08, y: y + 0.45, w: 1.79, h: 0.45, fontSize: 11.5, align: "center", color: last ? WHITE : MUT });
      if (i < 3) arrow(s, x + 1.98, y + 0.34, 0.34);
    });
  };
  row(1.75, "The forecast", [["20 frames", "measured velocity"], ["FNO model soup", "fine-tuned, weights averaged"], ["Turbulence fix", "restores lost fluctuations"], ["Forecast", "the next 20 frames"]], BLUE);
  row(3.25, "The uncertainty bands", [["Frames + forecast", "80 input channels"], ["U-Net", "expected error per point"], ["24-bin table", "error → band width"], ["Bands", "forecast ± width"]], OR);
}

// 8 Results vs baseline ---------------------------------------------------------------------------------------
{
  const s = slide("The entry raised the overall score from 72.7 to 80.1; uncertainty nearly tripled", { tag: P1, cite: "Development set; scores out of 100. Baseline: the organisers' FNO." });
  s.addImage({ path: IMG("d_dumbbell.png"), x: 0.4, y: 1.3, w: 6.3, h: 3.6 });
  stat(s, 6.9, 1.45, 2.7, "72.7 → 80.1", "overall score", OR, 30);
  stat(s, 6.9, 3.05, 2.7, "14 → 40", "uncertainty score", BLUE, 30);
}

// 9 The climb --------------------------------------------------------------------------------------------------
{
  const s = slide("The score climbed in small steps; adding uncertainty bands gave the largest jump", { tag: P1 });
  s.addImage({ path: IMG("d_climb.png"), x: 0.5, y: 1.35, w: 9, h: 3.2 });
  txt(s, "78.45 (first fine-tuned model)  →  80.08 (final entry)", { x: 0.5, y: 4.6, w: 9, h: 0.3, fontSize: 16, bold: true, align: "center" });
}

// 10 Where we stand --------------------------------------------------------------------------------------------
{
  const s = slide("We finished just outside the top 50; the top ten needed a better forecast, not better bands", { tag: P1, cite: "About 182 teams. Final cutoffs (17 Oct); the perfect-band ceiling was computed on 24 Sep." });
  s.addImage({ path: IMG("d_numberline.png"), x: 0.5, y: 1.35, w: 9, h: 2.6 });
  fillBox(s, 0.5, 4.1, 9, 0.75, PALE);
  txt(s, "Even perfect uncertainty bands on our forecast reach only 81.73, below the top-10 cutoff.", { x: 0.7, y: 4.1, w: 8.6, h: 0.75, fontSize: 17, bold: true, color: BLUE, valign: "middle", align: "center" });
}

// 11 What failed and why -------------------------------------------------------------------------------------
{
  const s = slide("What failed was just as informative: a 2-D physics rule is false for this data", { tag: P1 });
  [["−9 pts", "forcing the forecast to be divergence-free", OR], ["−16%", "U-Net, the benchmark's best model", BLUE], ["0", "gain from adding simulations to training", BLUE], ["+0.008", "a finer band table (24 → 4096 bins)", BLUE]].forEach(([b, l, c], i) => {
    const x = 0.5 + (i % 2) * 4.6, y = 1.4 + Math.floor(i / 2) * 1.25;
    txt(s, b, { x, y, w: 1.9, h: 1.0, fontSize: 30, bold: true, color: c, valign: "middle", align: "center" });
    txt(s, l, { x: x + 2.0, y, w: 2.5, h: 1.0, fontSize: 16, valign: "middle" });
  });
  fillBox(s, 0.5, 4.0, 9, 0.85, PALEO);
  txt(s, "The camera sees a 2-D slice of a 3-D flow: 45% of the measured fluctuation energy breaks the 2-D rule.", { x: 0.7, y: 4.0, w: 8.6, h: 0.85, fontSize: 16, bold: true, valign: "middle" });
}

// 12 Divider ---------------------------------------------------------------------------------------------------
divider("Part II", "What Does Pretraining Data Buy?", "Part I's open question, answered on a flow we control");

// 13 Research question ----------------------------------------------------------------------------------------
{
  const s = slide("Research question: which property of pretraining data produces the gain?", { tag: P2, cite: "Aurora: Bodnar et al., Nature 2025" });
  const box = (x, y, t, fill, dark) => { fillBox(s, x, y, 2.5, 0.8, fill); txt(s, t, { x: x + 0.1, y, w: 2.3, h: 0.8, fontSize: 15, valign: "middle", align: "center", color: dark ? WHITE : INK }); };
  txt(s, "Pretrained", { x: 0.5, y: 1.4, w: 3, h: 0.3, fontSize: 18, bold: true, color: BLUE });
  box(0.5, 1.8, "Pretrain on many related flow settings", PALE); arrow(s, 3.05, 2.06, 0.5);
  box(3.6, 1.8, "Fine-tune briefly on the target", PALE); arrow(s, 6.15, 2.06, 0.5);
  box(6.7, 1.8, "Predict unseen target flows", BLUE, true);
  txt(s, "From scratch", { x: 0.5, y: 2.85, w: 3, h: 0.3, fontSize: 18, bold: true, color: MUT });
  box(3.6, 3.25, "Train on the target only, same total steps", SOFT); arrow(s, 6.15, 3.51, 0.5);
  box(6.7, 3.25, "Predict the same unseen flows", GREY, true);
  txt(s, "Aurora reports a 25% gain from pretraining, but its data differ in amount, variety and closeness all at once.", { x: 0.5, y: 4.3, w: 9, h: 0.6, fontSize: 17 });
}

// 14 Kolmogorov flow ------------------------------------------------------------------------------------------
{
  const s = slide("Kolmogorov flow: a turbulent flow whose every property we control", { tag: P2, cite: "Kolmogorov flow: Boffetta & Ecke, 2012. Colour: red and blue are fluid spinning in opposite directions." });
  mathBox(s, "∂ω/∂t  +  u·∇ω  =  ν ∇²ω  −  α ω  +  f(y),        f ∝ cos(k_f y)", 1.3, { fontSize: 17 });
  [["kolmo0.png", "Re 100: calm", false], ["kolmo1.png", "Re 300: the target", true], ["kolmo2.png", "Re 1000: wilder", false]].forEach(([f, l, hi], i) => {
    const x = 0.5 + i * 2.1;
    s.addImage({ path: IMG(f), x, y: 1.95, w: 1.95, h: 1.95 });
    txt(s, l, { x, y: 3.95, w: 1.95, h: 0.3, fontSize: 14, bold: hi, color: hi ? OR : MUT, align: "center" });
  });
  txt(s, "Two knobs", { x: 7.0, y: 1.95, w: 2.5, h: 0.4, fontSize: 20, bold: true, color: BLUE });
  bullets(s, ["k_f: stirring bands (target 4)", "Re: how wild, set by viscosity ν (target 300)", "Target never seen in pretraining"], { x: 7.0, y: 2.45, w: 2.5, h: 2.4, fontSize: 15, paraSpaceAfter: 8 });
}

// 15 The design map -------------------------------------------------------------------------------------------
{
  const s = slide("Pretraining sets differ in size and variety; only the largest includes near-target settings", { tag: P2 });
  s.addImage({ path: IMG("d_map.png"), x: 0.4, y: 1.3, w: 5.6, h: 3.6 });
  bullets(s, [["2 or 8 settings:", "far from the target"], ["32 settings:", "includes its neighbours"], ["Each set at", "32, 64 or 128 simulations"]], { x: 6.3, y: 1.6, w: 3.2, h: 3.0 });
}

// 16 Toolkit + protocol ---------------------------------------------------------------------------------------
{
  const s = slide("We built a verified toolkit, and pre-registered every experiment", { tag: P2 });
  [["10⁻⁹", "simulator energy-balance error"], ["326", "simulations over 33 settings"], ["5.1 M", "parameters: Aurora rebuilt in JAX"], ["291", "training runs, 3 seeds each"]].forEach(([b, l], i) => stat(s, 0.5 + i * 2.3, 1.5, 2.1, b, l, i === 3 ? OR : BLUE, 38));
  fillBox(s, 0.5, 3.6, 9, 1.15, PALEO);
  txt(s, "Predictions were committed before any run", { x: 0.75, y: 3.7, w: 8.5, h: 0.45, fontSize: 20, bold: true, color: OR });
  txt(s, "Original experiments: 49 predictions made in advance, 25 hit. A miss stays a miss.", { x: 0.75, y: 4.18, w: 8.5, h: 0.5, fontSize: 16 });
}

// 17 Model ----------------------------------------------------------------------------------------------------
{
  const s = slide("Inside the model: patches, windowed attention, and a zoom-out, zoom-in backbone", { tag: P2, cite: "Aurora: Bodnar et al., 2025  |  Perceiver: Jaegle et al., 2021  |  Swin: Liu et al., 2021  |  LoRA: Hu E. et al., 2022" });
  const st = [["Two frames", "u, v, p, ω on a 64² grid"], ["Patches", "4 × 4 pixels give 256 tokens"], ["Perceiver", "5 frequency levels to 3"], ["Swin U-Net", "attention in windows, zoom out and in"], ["Next snapshot", "the change is added to the input"]];
  st.forEach(([h, d], i) => {
    const x = 0.5 + i * 1.88, end = i === 0 || i === 4;
    fillBox(s, x, 1.45, 1.62, 1.45, end ? BLUE : PALE);
    txt(s, h, { x: x + 0.08, y: 1.55, w: 1.46, h: 0.4, fontSize: 15, bold: true, color: end ? WHITE : INK, align: "center" });
    txt(s, d, { x: x + 0.08, y: 2.0, w: 1.46, h: 0.85, fontSize: 12, color: end ? WHITE : MUT, align: "center" });
    if (i < 4) arrow(s, x + 1.64, 2.05, 0.22);
  });
  bullets(s, ["5.1 M parameters; the core block matches Aurora's own code to 7 decimals"], { x: 0.5, y: 3.2, w: 9, h: 0.5, fontSize: 17 });
  mathBox(s, "LoRA fine-tuning:   W  →  W + B·A   (rank 4: trains only 0.84% of the weights)", 3.95);
}

// 18 Eleven experiments --------------------------------------------------------------------------------------
{
  const s = slide("Eleven experiments, one question each", { tag: P2 });
  const E = [["E1", "Amount vs variety", "same gain either way"], ["E2", "Mixed data formats", "simple padding beats the Perceiver"], ["E3", "Wilder flows", "still 36–41% better"],
    ["E4", "What parts matter", "pushforward steadies forecasts"], ["E5", "Scarce target data", "41% gain with one simulation"], ["E6", "Real measurements", "real data is what counts"],
    ["E7–9", "Attention upgrades", "about 9% better next steps"], ["E10", "What limits accuracy", "tiny swirls, the spin field"], ["E11", "Physics built in", "13% better 20-step forecasts"]];
  E.forEach(([id, q, a], i) => {
    const x = 0.5 + (i % 3) * 3.05, y = 1.3 + Math.floor(i / 3) * 1.0;
    fillBox(s, x, y, 2.9, 0.88, i === 8 || i === 0 ? PALEO : SOFT);
    txt(s, id, { x: x + 0.08, y, w: 0.85, h: 0.88, fontSize: 19, bold: true, color: i === 8 || i === 0 ? OR : BLUE, valign: "middle", align: "center" });
    txt(s, q, { x: x + 0.95, y: y + 0.1, w: 1.9, h: 0.32, fontSize: 13, bold: true });
    txt(s, a, { x: x + 0.95, y: y + 0.44, w: 1.9, h: 0.4, fontSize: 11.5, color: MUT });
  });
  mathBox(s, "error  e = ‖ forecast − truth ‖ / ‖ truth ‖              gain  =  1 − e(pretrained) / e(scratch)", 4.4, { fontSize: 15 });
}

// 19 Result 1 ------------------------------------------------------------------------------------------------
{
  const s = slide("Pretraining roughly halves the next-step error, from 0.150 to 0.081", { tag: P2, cite: "Next-step error on an unseen setting, 3 seeds. Flows: one step from the same moment (pretrained on 32 settings)." });
  s.addImage({ path: IMG("d_halves.png"), x: 0.4, y: 1.35, w: 4.6, h: 4.6 * 3.5 / 4.8 });
  [["c_truth.png", "truth", INK, true], ["c_pretrained.png", "pretrained: 0.19", BLUE, true], ["c_scratch.png", "scratch: 0.28", MUT, false]].forEach(([f, l, c, b], i) => {
    const x = 5.25 + i * 1.45;
    s.addImage({ path: IMG(f), x, y: 1.45, w: 1.35, h: 1.35 });
    txt(s, l, { x: x - 0.05, y: 2.85, w: 1.45, h: 0.3, fontSize: 12.5, bold: b, color: c, align: "center" });
  });
  txt(s, "spin error, one example", { x: 5.25, y: 3.15, w: 4.25, h: 0.3, fontSize: 11.5, color: MUT, align: "center", italic: true });
  bullets(s, ["42–50% lower in every setup", "Still 36–41% better on flows ~3× wilder"], { x: 5.4, y: 3.6, w: 4.1, h: 1.3, fontSize: 17, paraSpaceAfter: 6 });
}

// 20 Learning curves -----------------------------------------------------------------------------------------
{
  const s = slide("Pretrained models reach in 100 steps what scratch never reaches in 22,000", { tag: P2, cite: "Next-step error on held-out target trajectories during training; mean of 3 seeds." });
  s.addImage({ path: IMG("d_curves.png"), x: 0.4, y: 1.3, w: 5.6, h: 3.6 });
  bullets(s, [["Scratch:", "0.150 after 22,000 steps"], ["Pretrained:", "0.08 within 100 fine-tuning steps"], ["Close pretraining:", "already 0.08 at the start"]], { x: 6.3, y: 1.6, w: 3.2, h: 3.2 });
}

// 21 Result 2 ------------------------------------------------------------------------------------------------
{
  const s = slide("Closeness to the target matters most: near-target settings give the best start", { tag: P2, cite: "Error before any fine-tuning, by the number of pretraining settings." });
  s.addImage({ path: IMG("d_closeness.png"), x: 0.5, y: 1.35, w: 5.2, h: 3.5 });
  bullets(s, ["More data or variety: the same final gain", "Near-target settings: 0.084 before fine-tuning, against about 0.19"], { x: 6.1, y: 1.6, w: 3.4, h: 3.0 });
}

// 22 Result 3 ------------------------------------------------------------------------------------------------
{
  const s = slide("Pretraining pays most when target data is scarce: 41% gain with one simulation", { tag: P2, cite: "Gain = 1 − (pretrained error / scratch error), by target simulations available." });
  s.addImage({ path: IMG("d_scarce.png"), x: 0.5, y: 1.35, w: 5.4, h: 3.5 });
  txt(s, "41%", { x: 6.3, y: 1.5, w: 3.2, h: 0.75, fontSize: 44, bold: true, color: OR, valign: "middle" });
  txt(s, "gain with a single target simulation", { x: 6.3, y: 2.3, w: 3.2, h: 0.6, fontSize: 16, color: MUT });
  bullets(s, ["With 6: only 5%, scratch catches up", "Matches real experiments: data is expensive"], { x: 6.3, y: 3.2, w: 3.2, h: 1.6, fontSize: 17 });
}

// 23 Chaos ---------------------------------------------------------------------------------------------------
{
  const s = slide("Long forecasts are hard: small errors grow until the fine detail is lost", { tag: P2, cite: "One rollout of the normal model from a test frame; spin field. Error = relative error over all fields." });
  img(s, "d_chaos.png", 0.4, 1.3, 5.5, 3.6, "left");
  bullets(s, ["Each prediction feeds the next", "Error 0.26 → 0.84 → 1.17 by step 20", "So we also score 20- and 50-step forecasts"], { x: 6.1, y: 1.6, w: 3.4, h: 3.2, fontSize: 17 });
}

// 24 Result 4 ------------------------------------------------------------------------------------------------
{
  const s = slide("Pushforward is the biggest lever for long forecasts; attention and depth help next steps", { tag: P2, cite: "One change at a time against the base model, 3 seeds (E4, E7–E9, Stage 12)." });
  s.addImage({ path: IMG("d_components.png"), x: 0.4, y: 1.35, w: 6.0, h: 3.4 });
  bullets(s, [["Pushforward:", "train on the model's own predictions"], ["Not worth it here:", "the Perceiver (padding beats it)"], ["LoRA:", "fixes systematic errors, up to 71%"]], { x: 6.6, y: 1.5, w: 2.9, h: 3.4, fontSize: 15, paraSpaceAfter: 8 });
}

// 25 Result 5 ------------------------------------------------------------------------------------------------
{
  const s = slide("On real data, the rebuilt model is within one point of the organisers' model", { tag: P2, cite: "Held-out real recordings (RealPDE, Hu Z. et al., 2026); the FNO is fine-tuned on real data." });
  s.addImage({ path: IMG("d_realdata.png"), x: 0.5, y: 1.35, w: 6.0, h: 2.9 });
  fillBox(s, 0.5, 4.38, 9, 0.52, PALEO);
  txt(s, "Simulation pretraining adds only about 0.1: on real data, the real data is what counts.", { x: 0.7, y: 4.38, w: 8.6, h: 0.52, fontSize: 16, bold: true, valign: "middle", align: "center" });
  stat(s, 6.8, 1.6, 2.7, "2 vs 20", "past frames seen: ours vs the FNO", BLUE, 32);
}

// 26 Heatmap -------------------------------------------------------------------------------------------------
{
  const s = slide("The remaining error lives in the tiniest swirls; pretraining pushes that limit further", { tag: P2, cite: "E10: 30 trained models measured, no new training. Each cell: error power ÷ true power at that swirl size." });
  s.addImage({ path: IMG("d_heat.png"), x: 0.5, y: 1.4, w: 9, h: 2.9 });
  txt(s, "Big swirls: error under 1%.   Tiny swirls: error as large as the signal.", { x: 0.5, y: 4.45, w: 9, h: 0.35, fontSize: 16, bold: true, align: "center" });
}

// 27 Donuts --------------------------------------------------------------------------------------------------
{
  const s = slide("Two thirds of the error is in the spin field, and scratch memorises its data (32×)", { tag: P2, cite: "Share of the next-step error by field; pretrained = 32-setting models. Right: test error ÷ training error." });
  s.addImage({ path: IMG("d_donut.png"), x: 0.5, y: 1.4, w: 9, h: 3.3 });
}

// 28 The physics idea ----------------------------------------------------------------------------------------
{
  const s = slide("On this flow the spin fixes everything else, so the model can predict the spin alone", { tag: P2 });
  s.addImage({ path: IMG("d_fields.png"), x: 0.5, y: 1.3, w: 9, h: 3.3 });
  mathBox(s, "ψ = −∇⁻² ω      u = ∂ψ/∂y      v = −∂ψ/∂x      ∇²p = 2 (u_x v_y − u_y v_x)", 4.45, { fontSize: 16 });
}

// 29 Divergence maps -----------------------------------------------------------------------------------------
{
  const s = slide("The normal model breaks the no-leak rule by 12%; with physics built in, it never does", { tag: P2, cite: "Divergence ∂u/∂x + ∂v/∂y of one next-step prediction, relative to the velocity gradients (E11 models, seed 0)." });
  s.addImage({ path: IMG("d_div.png"), x: 0.5, y: 1.35, w: 9, h: 3.4 });
}

// 30 Physics result ------------------------------------------------------------------------------------------
{
  const s = slide("Building the physics in cuts the average 20-step forecast error by 13%", { tag: P2, cite: "Mean of 3 seeds and 84 rollouts per run; lower is better." });
  s.addImage({ path: IMG("d_rollout.png"), x: 0.4, y: 1.3, w: 5.4, h: 3.6 });
  txt(s, "−12.6%", { x: 6.1, y: 1.4, w: 3.4, h: 0.7, fontSize: 38, bold: true, color: OR, valign: "middle" });
  txt(s, "average error over 20 steps", { x: 6.1, y: 2.1, w: 3.4, h: 0.35, fontSize: 14, color: MUT });
  bullets(s, ["Best of any change tried here, pushforward included", "7% faster to train", "Pressure error −38%"], { x: 6.1, y: 2.65, w: 3.4, h: 2.2, fontSize: 16, paraSpaceAfter: 8 });
}

// 31 Together ------------------------------------------------------------------------------------------------
{
  const s = slide("Together: closeness, scarce target data and true physics are what help", { tag: "Synthesis" });
  bullets(s, [["Pretraining", "helps when it is close to the task and target data are scarce."], ["Far-away simulations", "did not help on the real wake."], ["Physics", "helps when it is true for the data: it hurt on camera data, helped on simulation."], ["Predictions written first", "turn surprises into results."]],
    { x: 0.5, y: 1.5, w: 9, h: 3.3, fontSize: 21, paraSpaceAfter: 16 });
}

// 32 Limits and next steps -----------------------------------------------------------------------------------
{
  const s = slide("Limits and next steps: claims are bounded by a small model and one flow family", { tag: "Discussion" });
  txt(s, "Limits", { x: 0.5, y: 1.45, w: 4.2, h: 0.4, fontSize: 22, bold: true, color: OR });
  bullets(s, ["5.1 M parameters, 64² fields, one flow family", "Fixed budgets, 3 seeds", "Closeness is confounded with distance", "Part I: just outside the top 50"], { x: 0.5, y: 2.0, w: 4.2, h: 2.9, fontSize: 17, paraSpaceAfter: 8 });
  txt(s, "Next steps", { x: 5.2, y: 1.45, w: 4.3, h: 0.4, fontSize: 22, bold: true, color: BLUE });
  bullets(s, ["Physics built in + pushforward + attention upgrades", "Loss weighted to small swirls", "Longer, ordered input history", "Larger model; better sim-to-real transfer"], { x: 5.2, y: 2.0, w: 4.3, h: 2.9, fontSize: 17, paraSpaceAfter: 8 });
}

// 33 Conclusions ---------------------------------------------------------------------------------------------
{
  const s = slide("Conclusions: pretraining helps when close and scarce; built-in physics steadies forecasts", { tag: "Conclusions" });
  bullets(s, [["Part I:", "decoded the score, calibrated uncertainty: 72.7 → 80.1."], ["Part II:", "pretraining halves the error; closeness and scarce target data matter most."], ["Physics built in:", "exact flow rules, 13% better 20-step forecasts."], ["On real data,", "the real data is what counts."]],
    { x: 0.5, y: 1.5, w: 9, h: 2.9, fontSize: 20, paraSpaceAfter: 12 });
  txt(s, "Code, reports and all results: github.com/ARYAMANN7279/aurora-from-scratch-jax", { x: 0.5, y: 4.6, w: 9, h: 0.35, fontSize: 15, bold: true, color: BLUE });
}

// 34 References ----------------------------------------------------------------------------------------------
{
  const s = slide("References", { tag: "Backup" });
  const refs = ["Li Z. et al. (2021). Fourier neural operator for parametric partial differential equations. ICLR.",
    "Hu Z. et al. (2026). RealPDEBench: a benchmark for complex physical systems with real-world data. ICLR.",
    "Wortsman M. et al. (2022). Model soups: averaging weights of fine-tuned models improves accuracy. ICML.",
    "Ronneberger O. et al. (2015). U-Net: networks for biomedical image segmentation. MICCAI.",
    "Bodnar C. et al. (2025). A foundation model for the Earth system. Nature 641, 1180–1187.",
    "Boffetta G., Ecke R. E. (2012). Two-dimensional turbulence. Annual Review of Fluid Mechanics 44, 427–451.",
    "Jaegle A. et al. (2021). Perceiver: general perception with iterative attention. ICML.",
    "Liu Z. et al. (2021). Swin Transformer: hierarchical vision transformer using shifted windows. ICCV.",
    "Hu E. J. et al. (2022). LoRA: low-rank adaptation of large language models. ICLR."];
  txt(s, refs.map((r, i) => ({ text: r, options: { breakLine: i < refs.length - 1 } })), { x: 0.5, y: 1.35, w: 9, h: 3.6, fontSize: 13, paraSpaceAfter: 7 });
}

// 35 Appendix A ----------------------------------------------------------------------------------------------
{
  const s = slide("Appendix A: how the competition score is computed", { tag: "Appendix" });
  const L = [["Each error channel:", "score(e) = 100 / (1 + 0.5 e)"], ["Speed:", "time = 100 / (1 + √(t / 729)),  t = run time in seconds"], ["Uncertainty:", "sps ≈ E[ e^(−w/σ) · 1(truth inside band) ],  σ = 0.056,  w = band width"],
    ["Development total:", "0.467 rel_l2 + 0.100 tke + 0.094 mvpe + 0.097 time + 0.247 sps + 0.912   (fit to 40 leaderboard rows, error under 0.005)"], ["Decision phase:", "a plain mean of the five channels (75.09 for our archive)"]];
  L.forEach(([h, d], i) => txt(s, [{ text: h + "  ", options: { bold: true, color: BLUE } }, { text: d }], { x: 0.5, y: 1.5 + i * 0.68, w: 9, h: 0.6, fontSize: 16 }));
}

// 36 Appendix B ----------------------------------------------------------------------------------------------
{
  const s = slide("Appendix B: from the predicted spin to velocity and pressure", { tag: "Appendix" });
  txt(s, "In Fourier space (wavenumber k):   ψ = ω / |k|²,   u = ∂ψ/∂y,   v = −∂ψ/∂x\nPressure:   ∇²p = 2 (u_x v_y − u_y v_x)", { x: 0.5, y: 1.35, w: 9, h: 0.8, fontSize: 15, paraSpaceAfter: 4 });
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK } } });
  const rows = [[hd("Variant (3 seeds)"), hd("One-step error"), hd("20-step drift"), hd("50-step drift")],
    ["Normal model", "0.150", "0.745", "1.100"], ["Spin only, loss on all four fields", "0.145 (−3.7%)", "0.651 (−12.6%)", "1.043 (−5.2%)"], ["Spin only, loss on the spin", "0.137 (−8.7%)", "0.709 (−4.7%)", "1.206 (+9.8%)"]];
  s.addTable(rows, { x: 0.5, y: 2.3, w: 9, colW: [3.6, 1.8, 1.8, 1.8], fontFace: FONT, fontSize: 14, color: INK, border: { type: "solid", pt: 0.5, color: LIGHT }, rowH: 0.45, valign: "middle", margin: [0, 8, 0, 8] });
  txt(s, "Divergence-free and curl-consistent to float32 round-off; trained 7% faster.", { x: 0.5, y: 4.4, w: 9, h: 0.4, fontSize: 14, color: MUT });
}

// 37 Appendix C ----------------------------------------------------------------------------------------------
{
  const s = slide("Appendix C: the prediction record", { tag: "Appendix" });
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK } } });
  const R = [["E1 volume vs variety", "2 / 6"], ["E2 mixed formats, Perceiver", "1 / 3"], ["E3 wilder flows", "3 / 4"], ["E4 model ablations", "1 / 5"], ["E5 target data, LoRA", "2 / 4"], ["E6 sim-to-real", "2 / 5"],
    ["E7–E9 attention", "11 / 19"], ["E10 diagnostics", "3 / 6"], ["E11 physics built in", "3 / 6"]];
  const half = (rs) => [[hd("Experiment"), hd("Hit")]].concat(rs.map((r) => [r[0], { text: r[1], options: { align: "right" } }]));
  const o = (x) => ({ x, y: 1.45, w: 4.3, colW: [3.2, 1.1], fontFace: FONT, fontSize: 14, color: INK, border: { type: "solid", pt: 0.5, color: LIGHT }, rowH: 0.42, valign: "middle", margin: [0, 8, 0, 8] });
  s.addTable(half(R.slice(0, 5)), o(0.5));
  s.addTable(half(R.slice(5)), o(5.2));
  txt(s, "Original experiments (E1–E9): 25 of 49 predictions hit. A miss is not a failed experiment; it means the expectation was wrong.", { x: 0.5, y: 4.3, w: 9, h: 0.6, fontSize: 14, color: MUT });
}

// 38 Appendix D ----------------------------------------------------------------------------------------------
{
  const s = slide("Appendix D: glossary", { tag: "Appendix" });
  const hd = (t) => ({ text: t, options: { bold: true, color: WHITE, fill: { color: INK } } });
  const G = [["Reynolds number (Re)", "how wild (turbulent) a flow is; higher is wilder"], ["Vorticity (spin), ω", "how fast the fluid rotates at each point"], ["PIV", "particle image velocimetry: cameras track particles to measure velocity"],
    ["Pretraining / fine-tuning", "learning from lots of related data, then briefly on the task"], ["Zero-shot", "using the pretrained model with no fine-tuning"], ["Rollout", "feeding predictions back in to forecast many steps"],
    ["Pushforward", "also training on the model's own predictions, so rollouts stay steady"], ["LoRA", "fine-tuning a tiny add-on instead of all the weights"]];
  s.addTable([[hd("Term"), hd("Meaning")]].concat(G), { x: 0.5, y: 1.3, w: 9, colW: [2.6, 6.4], fontFace: FONT, fontSize: 13, color: INK, border: { type: "solid", pt: 0.5, color: LIGHT }, rowH: 0.38, valign: "middle", margin: [0, 8, 0, 8] });
}

if (n !== NOTES.length) throw new Error(`slides ${n} != notes ${NOTES.length}`);
const words = NOTES.map((t) => t.split(/\s+/).length);
const talk = words.slice(0, 33).reduce((a, b) => a + b, 0);
let md = "# UGP Parts I and II: speaker notes\n\n";
md += `Slides 1–33 are the talk, about 30 seconds each (${Math.round(talk / 150)} to ${Math.round(talk / 130)} minutes at a calm pace); slides 34–38 are backup for questions.\n`;
NOTES.forEach((t, i) => { md += `\n## ${i + 1}. ${TITLES[i]}\n\n${t}\n`; });
fs.writeFileSync(path.join(__dirname, "..", "UGP_Final_speaker_notes.md"), md);
console.log("max words in a note:", Math.max(...words), "talk words:", talk);
pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f, n, "slides"));
