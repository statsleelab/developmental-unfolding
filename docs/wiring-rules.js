"use strict";

const byId = (id) => document.getElementById(id);
const splitSelect = byId("wiring-split");
const score = new Intl.NumberFormat("en", { minimumFractionDigits: 4, maximumFractionDigits: 4 }).format;
const delta = new Intl.NumberFormat("en", {
  minimumFractionDigits: 4, maximumFractionDigits: 4, signDisplay: "exceptZero",
}).format;
const count = new Intl.NumberFormat("en").format;

const models = [
  ["baseline", "Contact and type"],
  ["named_main", "Add individual gene terms"],
  ["named_pairs_5", "Add 5 gene pairs"],
  ["named_pairs_10", "Add 10 gene pairs"],
  ["named_pairs_20", "Add 20 gene pairs"],
];

const splits = {
  held_specimen: {
    label: "Hold out one specimen",
    description: "Test a full specimen that the model did not see during fitting.",
  },
  held_specimen_and_subtype_pair: {
    label: "Hold out specimen and subtype pairs",
    description: "Also exclude the tested subtype pairs in both directions from training.",
  },
  early_to_later: {
    label: "Train early, test later",
    description: "Fit each model once on Dataset1–3 and test on Dataset4, 5, 6, and 8. Use the same late-embryonic expression profiles and the observed anatomy of each specimen.",
  },
};

function element(tag, value) {
  const node = document.createElement(tag);
  node.textContent = value;
  return node;
}

function tableRow(label, values) {
  const row = document.createElement("tr");
  const heading = element("th", label);
  heading.scope = "row";
  row.append(heading, ...values.map((value) => element("td", value)));
  return row;
}

function definitionList(id, entries) {
  byId(id).replaceChildren(...entries.map(([label, value]) => {
    const row = document.createElement("div");
    row.append(element("dt", label), element("dd", value));
    return row;
  }));
}

function showSplit(results, key) {
  const split = results[key];
  const named = split.mean.named_pairs_10;
  byId("split-description").textContent = `${splits[key].description} ${count(split.specimens.length)} test specimens.`;
  byId("wiring-scores").replaceChildren(...models.map(([model, label]) => {
    const { ap, log_loss } = split.mean[model];
    return tableRow(label, [score(ap), score(log_loss)]);
  }));
  byId("specimen-scores").replaceChildren(...split.specimens.map(({ specimen, scores }) => {
    const reference = scores.baseline;
    const interaction = scores.named_pairs_10;
    return tableRow(specimen, [reference.ap, interaction.ap, reference.log_loss, interaction.log_loss].map(score));
  }));
  for (const [id, other] of [["baseline-contrast", split.mean.baseline], ["main-contrast", split.mean.named_main]]) {
    definitionList(id, [
      ["Average precision", delta(named.ap - other.ap)],
      ["Log loss", delta(named.log_loss - other.log_loss)],
    ]);
  }
}

function showRules(data) {
  const folds = data.results.held_specimen.specimens.length;
  definitionList("panel-summary", [
    ["Genes used", data.panel.used.join(", ")],
    ["Panel genes omitted", data.panel.dropped.join(", ") || "None"],
    ["Candidate gene-pair products", count(data.panel.used.length ** 2)],
    ["Directed contact observations", count(data.candidate_count)],
  ]);
  byId("rules-caption").textContent = `Selection and coefficient signs across ${count(folds)} specimen-only folds.`;
  byId("wiring-rules").replaceChildren(...data.rules.map((rule) => tableRow(
    `${rule.source_gene} × ${rule.target_gene}`,
    [`${rule.selected_folds} of ${folds}`, rule.positive_folds, rule.negative_folds],
  )));
}

function showMotifs(motifs) {
  const rows = [
    ["reciprocal", "Reciprocal pair"],
    ["divergent", "Divergent pair"],
    ["convergent", "Convergent pair"],
    ["feedforward", "Feedforward triangle"],
  ];
  byId("wiring-motifs").replaceChildren(...rows.map(([name, label]) => tableRow(
    label,
    [motifs.observed[name], motifs.baseline[name], motifs.named_pairs_10[name]].map(count),
  )));
}

async function loadWiringRules() {
  const status = byId("wiring-status");
  try {
    const response = await fetch("./wiring-rules-results.json");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    showRules(data);
    showMotifs(data.adult_motifs);
    for (const [scheme, { label }] of Object.entries(splits)) {
      if (!data.results[scheme]) continue;
      const option = element("option", label);
      option.value = scheme;
      splitSelect.append(option);
    }
    showSplit(data.results, splitSelect.value);
    splitSelect.addEventListener("change", () => showSplit(data.results, splitSelect.value));
    for (const id of ["prediction-content", "rules-content", "motifs-content"]) byId(id).hidden = false;
    status.hidden = true;
  } catch (error) {
    status.textContent = "Wiring-rule results could not be loaded. Reload the page to try again.";
    console.error(error);
  }
}

void loadWiringRules();
