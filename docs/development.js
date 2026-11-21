"use strict";

const byId = (id) => document.getElementById(id);
const canvas = byId("growth-canvas");
const slider = byId("growth-slider");
const context = canvas.getContext("2d");
const number = new Intl.NumberFormat("en", { maximumFractionDigits: 1 });
const percentage = new Intl.NumberFormat("en", { style: "percent", maximumFractionDigits: 2 }).format;

function percentagePoints(value) {
  const points = Math.round(value * 10000) / 100;
  return (points > 0 ? "+" : "") + points.toFixed(2).replace("-", "−") + " pp";
}

function intervalText(low, high, format) {
  return `[${format(low)}, ${format(high)}]`;
}

function comparisonText(interval, positive, negative) {
  if (interval.low > 0) return positive;
  if (interval.high < 0) return negative;
  return "The paired interval includes zero.";
}

function makeElement(tag, text, className) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text) element.textContent = text;
  return element;
}

function writeList(id, entries) {
  byId(id).replaceChildren(...entries.map(([label, value, detail]) => {
    const row = makeElement("div");
    const description = makeElement("dd", value);
    if (detail) description.append(makeElement("span", detail));
    row.append(makeElement("dt", label), description);
    return row;
  }));
}

function writeResults(data) {
  const { config, evaluation } = data;
  const { structure, learning, diagnostics } = evaluation;
  const structureLabels = [
    ["Neighbor signaling", structure.growth],
    ["Signaling disabled", structure.signal_off],
    ["Signaling effect", structure.difference],
  ];
  writeList("structure-values", structureLabels.map(([label, interval]) => [
    label,
    percentagePoints(interval.mean),
    intervalText(interval.low, interval.high, percentagePoints),
  ]));
  byId("structure-interpretation").textContent = comparisonText(
    structure.difference,
    "The paired interval supports greater spatial clustering with neighbor signaling.",
    "The paired interval supports greater spatial clustering with signaling disabled.",
  );

  writeList("development-diagnostics", [
    ["Lattice", `${config.grid_size} × ${config.grid_size}`],
    ["Development ticks", String(config.ticks)],
    ["Mean cells", number.format(diagnostics.cells)],
    ["Mean differentiated fraction", percentage(diagnostics.mature_fraction)],
    ["Mean excitatory share of differentiated cells", percentage(diagnostics.excitatory_fraction)],
    ["Mean connections", number.format(diagnostics.edges)],
  ]);
  byId("evaluation-count").textContent = `${evaluation.seeds.length} held-out growth seeds`;

  const contrast = learning.auc_difference;
  byId("learning-difference").textContent = percentagePoints(contrast.mean);
  byId("learning-interval").textContent = `95% interval ${intervalText(contrast.low, contrast.high, percentagePoints)}`;
  byId("learning-interpretation").textContent = comparisonText(
    contrast,
    "The paired interval favors growth wiring across the tested budgets.",
    "The paired interval favors endpoint wiring across the tested budgets.",
  );
  writeList("learning-diagnostics", [
    ["Delay", `${config.delay} steps`],
    ["Growth connections replaced by control", percentage(diagnostics.replaced_fraction)],
  ]);
  if (diagnostics.uncoupled_accuracy?.length) {
    const accuracies = diagnostics.uncoupled_accuracy;
    const bounds = `${percentage(Math.min(...accuracies))}–${percentage(Math.max(...accuracies))}`;
    const note = byId("uncoupled-note");
    note.textContent = `With recurrent connections removed, mean accuracy ranged from ${bounds} across budgets. Chance is 50%.`;
    note.hidden = false;
  }

  const curves = [learning.curves.growth, learning.curves.endpoint];
  byId("learning-rows").replaceChildren(...config.budgets.map((budget, index) => {
    const row = makeElement("tr");
    const heading = makeElement("th", String(budget));
    heading.scope = "row";
    row.append(heading);
    for (const curve of curves) {
      const cell = makeElement("td", percentage(curve.mean[index]));
      cell.append(makeElement("span", intervalText(curve.low[index], curve.high[index], percentage)));
      row.append(cell);
    }
    return row;
  }));
}

function viewport(frames) {
  let left = Infinity;
  let right = -Infinity;
  let top = Infinity;
  let bottom = -Infinity;
  for (const frame of frames) {
    for (const [x, y] of frame.cells) {
      left = Math.min(left, x);
      right = Math.max(right, x);
      top = Math.min(top, y);
      bottom = Math.max(bottom, y);
    }
  }
  return {
    x: (left + right) / 2,
    y: (top + bottom) / 2,
    side: Math.max(right - left, bottom - top) + 3,
  };
}

function drawColony(frame, bounds, colors) {
  if (!context || !canvas.clientWidth) return;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const pixelRatio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

  const step = Math.min(width, height) / bounds.side;
  const xAt = (x) => width / 2 + (x - bounds.x) * step;
  const yAt = (y) => height / 2 + (y - bounds.y) * step;
  const side = Math.max(2, step * 0.38);
  const start = Math.ceil(-xAt(0) / step);
  const end = Math.floor((width - xAt(0)) / step);
  context.strokeStyle = "#dde3d7";
  context.lineWidth = 1;
  context.beginPath();
  for (let coordinate = start; coordinate <= end; coordinate++) {
    context.moveTo(xAt(coordinate) - step / 2, 0);
    context.lineTo(xAt(coordinate) - step / 2, height);
  }
  const firstRow = Math.ceil(-yAt(0) / step);
  const lastRow = Math.floor((height - yAt(0)) / step);
  for (let coordinate = firstRow; coordinate <= lastRow; coordinate++) {
    context.moveTo(0, yAt(coordinate) - step / 2);
    context.lineTo(width, yAt(coordinate) - step / 2);
  }
  context.stroke();

  for (const [x, y, fate] of frame.cells) {
    const left = xAt(x);
    const top = yAt(y);
    context.beginPath();
    context.fillStyle = colors[fate];
    if (fate === -1) {
      context.moveTo(left, top - side);
      context.lineTo(left + side, top);
      context.lineTo(left, top + side);
      context.lineTo(left - side, top);
      context.closePath();
      context.fill();
    } else {
      context.arc(left, top, side, 0, 2 * Math.PI);
      if (fate === 0) {
        context.strokeStyle = colors[0];
        context.lineWidth = Math.max(1, step * 0.12);
        context.stroke();
      } else {
        context.fill();
      }
    }
  }
}

function activateGrowth(example) {
  const { seed, frames } = example;
  const bounds = viewport(frames);
  const style = getComputedStyle(canvas);
  const colors = {
    "-1": style.getPropertyValue("--inhibitory").trim(),
    0: style.getPropertyValue("--uncommitted").trim(),
    1: style.getPropertyValue("--excitatory").trim(),
  };

  function update() {
    const frame = frames[Number(slider.value)];
    const count = { "-1": 0, 0: 0, 1: 0 };
    for (const [, , fate] of frame.cells) count[fate]++;
    const cellLabel = frame.cells.length === 1 ? "cell" : "cells";
    const connectionLabel = frame.edges === 1 ? "connection" : "connections";
    byId("growth-tick").value = String(frame.tick);
    slider.setAttribute("aria-valuetext", `Tick ${frame.tick}`);
    byId("growth-caption").textContent = `Seed ${seed}, tick ${frame.tick}. ${frame.cells.length} ${cellLabel} (${count[0]} uncommitted, ${count[1]} excitatory, ${count[-1]} inhibitory). ${frame.edges} directed ${connectionLabel}.`;
    drawColony(frame, bounds, colors);
  }

  slider.max = String(frames.length - 1);
  slider.value = slider.max;
  slider.addEventListener("input", update);
  update();
  if ("ResizeObserver" in window) {
    new ResizeObserver(() => drawColony(frames[Number(slider.value)], bounds, colors)).observe(canvas);
  } else {
    window.addEventListener("resize", () => drawColony(frames[Number(slider.value)], bounds, colors));
  }
}

async function loadResults() {
  const status = byId("data-status");
  const development = byId("development-content");
  const learning = byId("learning-content");
  try {
    const response = await fetch("./development-results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (!data.example?.frames?.length || !data.example.frames.every((frame) => frame.cells.length)) {
      throw new Error("Missing growth frames");
    }
    writeResults(data);
    development.hidden = false;
    learning.hidden = false;
    activateGrowth(data.example);
    status.hidden = true;
  } catch (error) {
    status.textContent = "Developmental results could not be loaded. Reload the page to try again.";
    development.hidden = true;
    learning.hidden = true;
    console.error(error);
  }
}

void loadResults();
