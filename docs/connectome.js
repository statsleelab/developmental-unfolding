"use strict";

const byId = (id) => document.getElementById(id);
const labels = { contact: "Physical contact", type: "Contact + cell type", expression: "Contact + cell type + expression" };

function render(summary) {
  const scheme = summary.results[byId("scheme").value];
  const rows = Object.entries(labels).map(([key, label]) => {
    const row = document.createElement("tr");
    const title = document.createElement("th");
    title.scope = "row";
    title.textContent = label;
    row.append(title);
    for (const value of [scheme.mean[key].ap, scheme.mean[key].log_loss]) {
      const score = document.createElement("td");
      score.textContent = value.toFixed(3);
      row.append(score);
    }
    return row;
  });
  byId("scores").replaceChildren(...rows);
  const difference = scheme.mean.expression.ap - scheme.mean.type.ap;
  byId("finding").textContent = `Expression changed mean average precision by ${difference >= 0 ? "+" : ""}${difference.toFixed(3)}.`;
}

async function loadResults() {
  try {
    const response = await fetch("./connectome-results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const summary = await response.json();
    const count = summary.coverage.reduce((total, specimen) => total + specimen.retained_candidates, 0);
    byId("coverage").textContent = `${summary.coverage.length} specimens · ${count.toLocaleString()} directed contact candidates · ${summary.expression_profiles} embryonic subtype profiles`;
    render(summary);
    byId("scheme").addEventListener("change", () => render(summary));
    byId("status").hidden = true;
    byId("metrics").hidden = false;
  } catch (error) {
    byId("status").textContent = "Results could not be loaded.";
    console.error(error);
  }
}

void loadResults();
