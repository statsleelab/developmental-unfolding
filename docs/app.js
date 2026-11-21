"use strict";

const byId = (id) => document.getElementById(id);

const series = [
  { key: "activity", label: "Activity pruning", color: "#24725e", band: "#24725e1b", dash: [] },
  { key: "random", label: "Matched random", color: "#a0522d", band: "#a0522d18", dash: [7, 4] },
  { key: "rewired", label: "Matched rewiring", color: "#6357a0", band: "#6357a01c", dash: [2, 4] },
];
const phases = {
  evaluation: { title: "Held-out evaluation", explanation: "Exploratory results on growth seeds reserved for evaluation." },
  pilot: { title: "Pilot", explanation: "Development runs used to inspect the protocol and check feasibility." },
};
const contrastLabels = {
  activity_minus_random: "Activity pruning − matched random",
  activity_minus_rewired: "Activity pruning − matched rewiring",
};

const canvas = byId("memory-chart");
const slider = byId("delay-slider");
const context = canvas.getContext("2d");
let data;
let phase = "evaluation";
let delayIndex = 0;
let plot;

function element(tag, value, className) {
  const node = document.createElement(tag);
  node.textContent = value;
  if (className) node.className = className;
  return node;
}

function decimal(value) {
  return (Math.abs(value) < 0.0005 ? 0 : value).toFixed(3).replace("-", "−");
}

function signed(value) {
  return value >= 0.0005 ? `+${decimal(value)}` : decimal(value);
}

function percent(value) {
  return new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 1 }).format(value);
}

function interval(low, high) {
  return `[${decimal(low)}, ${decimal(high)}]`;
}

function entries(id, pairs) {
  byId(id).replaceChildren(...pairs.map(([label, value]) => {
    const row = document.createElement("div");
    row.append(element("dt", label), element("dd", value));
    return row;
  }));
}

function renderConfiguration() {
  const config = data.config;
  entries("config-values", [
    ["Cells per circuit", config.nodes],
    ["Maturation steps", config.development_steps],
    ["Retention quota per source and distance", percent(config.kept_fraction)],
    ["Input steps per split", config.split_steps],
    ["Washout steps", config.washout],
    ["Delays tested", `${config.delays.at(0)}–${config.delays.at(-1)} steps (${config.delays.length})`],
  ]);
}

function renderDiagnostics(diagnostics) {
  entries("diagnostic-values", [
    ["Mean retained edges", diagnostics.edge_count.toFixed(1)],
    ["Mean total wiring length", diagnostics.wiring_length.toFixed(1)],
    ["Activity edges replaced by random pruning", percent(diagnostics.random_coverage)],
    ["Activity edges replaced by rewiring", percent(diagnostics.rewire_coverage)],
    ["Rewiring proposals accepted", percent(diagnostics.rewire_acceptance)],
    ["Maximum convergence ratio after washout", diagnostics.stability_max.toExponential(2)],
  ]);
}

function renderContrasts(summary) {
  const cards = Object.entries(contrastLabels).map(([key, label]) => {
    const { mean, low, high } = summary.contrasts[key];
    let comment = "Interval includes zero.";
    if (low > 0) comment = "Interval lies above zero.";
    if (high < 0) comment = "Interval lies below zero.";
    const card = document.createElement("div");
    card.className = "contrast-item";
    card.append(
      element("h4", label),
      element("p", signed(mean), "contrast-mean"),
      element("p", `95% CI ${interval(low, high)}`, "contrast-interval"),
      element("p", comment, "contrast-comment"),
    );
    return card;
  });
  byId("contrast-values").replaceChildren(...cards);
}

function renderTable(summary) {
  const rows = data.config.delays.map((delay, index) => {
    const row = document.createElement("tr");
    const title = element("th", delay);
    title.scope = "row";
    row.append(title);
    for (const { key } of series) {
      const { mean, low, high } = summary.curves[key];
      row.append(element("td", `${decimal(mean[index])} ${interval(low[index], high[index])}`));
    }
    return row;
  });
  byId("curve-rows").replaceChildren(...rows);
}

function selectDelay(index) {
  delayIndex = index;
  const delay = data.config.delays[index];
  slider.value = index;
  slider.setAttribute("aria-valuetext", `${delay} ${delay === 1 ? "step" : "steps"}`);
  byId("delay-value").value = `${delay} ${delay === 1 ? "step" : "steps"}`;
  const scores = series.map(({ key, label }) => {
    const { mean, low, high } = data[phase].curves[key];
    const cell = document.createElement("div");
    cell.className = "delay-score";
    cell.append(
      element("span", label),
      element("strong", decimal(mean[index])),
      element("span", interval(low[index], high[index]), "delay-interval"),
    );
    return cell;
  });
  byId("delay-scores").replaceChildren(...scores);
  drawChart();
}

function selectPhase(nextPhase) {
  if (!data) return;
  phase = nextPhase;
  const summary = data[phase];
  const seedCount = summary.seeds.length;
  const seedLabel = seedCount === 1 ? "growth seed" : "growth seeds";
  const intervalsIncludeZero = Object.values(summary.contrasts).every(({ low, high }) => low <= 0 && high >= 0);
  const outcome = intervalsIncludeZero ? "Both paired 95% intervals include zero." : "See the paired intervals for each comparison.";
  byId("experiment-note").textContent = `${seedCount} ${seedLabel}. ${phases[phase].explanation} ${outcome}`;
  byId("diagnostic-note").textContent = `${phases[phase].title} phase. The convergence ratio compares state differences after washout with their initial size.`;
  byId("evaluation-button").setAttribute("aria-pressed", phase === "evaluation");
  byId("pilot-button").setAttribute("aria-pressed", phase === "pilot");
  renderContrasts(summary);
  renderDiagnostics(summary.diagnostics);
  renderTable(summary);
  selectDelay(delayIndex);
}

function yAxis() {
  let low = 0;
  let high = 0;
  for (const phaseKey of Object.keys(phases)) {
    for (const { key } of series) {
      const curve = data[phaseKey].curves[key];
      low = Math.min(low, ...curve.low, ...curve.mean);
      high = Math.max(high, ...curve.high, ...curve.mean);
    }
  }
  const roughStep = Math.max(0.005, (high - low) / 4);
  const scale = 10 ** Math.floor(Math.log10(roughStep));
  const unit = [1, 2, 2.5, 5, 10].find((value) => value * scale >= roughStep);
  const step = unit * scale;
  const minimum = Math.floor(low / step) * step;
  const maximum = Math.ceil(high / step) * step || step;
  return { minimum, maximum, step };
}

function drawChart() {
  if (!data || !context || canvas.clientWidth === 0) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const density = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * density);
  canvas.height = Math.round(height * density);
  context.setTransform(density, 0, 0, density, 0, 0);

  const delays = data.config.delays;
  const curves = data[phase].curves;
  const axis = yAxis();
  const range = axis.maximum - axis.minimum;
  plot = { left: 48, right: width - 12, top: 12, bottom: height - 44 };
  const firstDelay = delays[0];
  const span = delays.at(-1) - firstDelay;
  const toX = (delay) => span ? plot.left + (delay - firstDelay) / span * (plot.right - plot.left) : (plot.left + plot.right) / 2;
  const toY = (score) => plot.bottom - (score - axis.minimum) / range * (plot.bottom - plot.top);
  const pointX = (index) => toX(delays[index]);

  context.font = "11px system-ui, sans-serif";
  context.fillStyle = "#4f6057";
  context.textAlign = "right";
  context.textBaseline = "middle";
  const ylabelDigits = Number(axis.step.toPrecision(2)).toString().split(".")[1]?.length ?? 0;
  const tickCount = Math.round(range / axis.step);
  for (let i = 0; i <= tickCount; i++) {
    const tick = axis.minimum + axis.step * i;
    const y = toY(tick);
    const isZero = Math.abs(tick) < 1e-10;
    context.strokeStyle = isZero ? "#a7b3a7" : "#dce0d6";
    context.beginPath();
    context.moveTo(plot.left, y);
    context.lineTo(plot.right, y);
    context.stroke();
    context.fillText((isZero ? 0 : tick).toFixed(ylabelDigits).replace("-", "−"), plot.left - 7, y);
  }

  context.textAlign = "center";
  const xTickCount = Math.min(delays.length, width < 400 ? 5 : 8);
  for (let i = 0; i < xTickCount; i++) {
    const index = Math.round(i * (delays.length - 1) / Math.max(1, xTickCount - 1));
    context.fillText(delays[index], pointX(index), plot.bottom + 13);
  }
  context.fillText("Delay (steps)", (plot.left + plot.right) / 2, height - 5);
  context.save();
  context.translate(8, (plot.top + plot.bottom) / 2);
  context.rotate(-Math.PI / 2);
  context.fillText("Held-out R²", 0, 0);
  context.restore();

  for (const { key, band } of series) {
    const { low, high } = curves[key];
    context.fillStyle = band;
    context.beginPath();
    context.moveTo(pointX(0), toY(high[0]));
    for (let i = 1; i < delays.length; i++) context.lineTo(pointX(i), toY(high[i]));
    for (let i = delays.length - 1; i >= 0; i--) context.lineTo(pointX(i), toY(low[i]));
    context.closePath();
    context.fill();
  }

  const selectedX = pointX(delayIndex);
  context.strokeStyle = "#687b72";
  context.setLineDash([3, 4]);
  context.beginPath();
  context.moveTo(selectedX, plot.top);
  context.lineTo(selectedX, plot.bottom);
  context.stroke();

  for (const { key, color, dash } of series) {
    const { mean } = curves[key];
    context.strokeStyle = color;
    context.lineWidth = 2.1;
    context.setLineDash(dash);
    context.beginPath();
    context.moveTo(pointX(0), toY(mean[0]));
    for (let i = 1; i < delays.length; i++) context.lineTo(pointX(i), toY(mean[i]));
    context.stroke();
  }

  context.setLineDash([]);
  for (const { key, color } of series) {
    context.fillStyle = color;
    context.strokeStyle = "#fdfbf5";
    context.lineWidth = 1.5;
    context.beginPath();
    context.arc(selectedX, toY(curves[key].mean[delayIndex]), 4.2, 0, Math.PI * 2);
    context.fill();
    context.stroke();
  }
}

function isComplete(result) {
  const config = result?.config;
  if (!Array.isArray(config?.delays) || config.delays.length === 0 || !config.delays.every(Number.isFinite)) return false;
  const configKeys = ["nodes", "development_steps", "kept_fraction", "split_steps", "washout"];
  if (!configKeys.every((key) => Number.isFinite(config[key]))) return false;
  return Object.keys(phases).every((key) => {
    const summary = result[key];
    const diagnosticsKeys = ["edge_count", "wiring_length", "random_coverage", "rewire_coverage", "rewire_acceptance", "stability_max"];
    return Array.isArray(summary?.seeds) && summary.seeds.length > 0 &&
      diagnosticsKeys.every((field) => Number.isFinite(summary.diagnostics?.[field])) &&
      Object.keys(contrastLabels).every((field) => ["mean", "low", "high"].every((name) => Number.isFinite(summary.contrasts?.[field]?.[name]))) &&
      series.every(({ key: curve }) => ["mean", "low", "high"].every((name) => {
        const values = summary.curves?.[curve]?.[name];
        return Array.isArray(values) && values.length === config.delays.length && values.every(Number.isFinite);
      }));
  });
}

async function start() {
  const status = byId("data-status");
  try {
    const response = await fetch("results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const result = await response.json();
    if (!isComplete(result)) throw new Error("Unexpected experiment data format.");
    data = result;
    slider.max = data.config.delays.length - 1;
    byId("results-content").hidden = false;
    renderConfiguration();
    selectPhase(phase);
    status.hidden = true;
    if ("ResizeObserver" in window) new ResizeObserver(drawChart).observe(canvas);
    else window.addEventListener("resize", drawChart);
  } catch (error) {
    status.textContent = "Experiment results could not be loaded. Reload the page to try again.";
    console.error(error);
  }
}

byId("evaluation-button").addEventListener("click", () => selectPhase("evaluation"));
byId("pilot-button").addEventListener("click", () => selectPhase("pilot"));
slider.addEventListener("input", () => selectDelay(Number(slider.value)));
canvas.addEventListener("pointerdown", (event) => {
  if (!plot) return;
  const delays = data.config.delays;
  const x = event.clientX - canvas.getBoundingClientRect().left;
  const fraction = (x - plot.left) / (plot.right - plot.left);
  const target = delays[0] + fraction * (delays.at(-1) - delays[0]);
  let closest = 0;
  for (let i = 1; i < delays.length; i++) {
    if (Math.abs(delays[i] - target) < Math.abs(delays[closest] - target)) closest = i;
  }
  selectDelay(closest);
});

void start();
