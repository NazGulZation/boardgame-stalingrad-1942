"use strict";

/* Stalingrad 1942 - frontend. Renders server state, sends actions to API. */

let state = null;
let selectedId = null;
let moves = [];
let targets = [];
let busy = false;
let aiRunning = false;

const boardEl = document.getElementById("board");
const bannerEl = document.getElementById("banner");
const unitDetailsEl = document.getElementById("unit-details");
const objectiveListEl = document.getElementById("objective-list");
const logListEl = document.getElementById("log-list");
const messageEl = document.getElementById("message");
const overlayEl = document.getElementById("overlay");
const overlayTitleEl = document.getElementById("overlay-title");
const overlayTextEl = document.getElementById("overlay-text");
const aiAxisEl = document.getElementById("ai-axis");
const aiSovietEl = document.getElementById("ai-soviet");
const modelSelectAxisEl = document.getElementById("model-select-axis");
const modelSelectSovietEl = document.getElementById("model-select-soviet");
const wrapModelAxisEl = document.getElementById("wrap-model-axis");
const wrapModelSovietEl = document.getElementById("wrap-model-soviet");
const UNIT_ICONS = { rifle: "R", sniper: "S", tank: "T" };

async function api(path, body) {
  let res;
  try {
    res = await fetch(path, {
      method: body ? "POST" : "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (err) {
    showMessage("Cannot reach the server.");
    return null;
  }
  let data;
  try {
    data = await res.json();
  } catch (err) {
    showMessage("Bad server response.");
    return null;
  }
  if (!res.ok) {
    showMessage(data.error || "Request failed.");
    return null;
  }
  return data;
}

let messageTimer = null;
function showMessage(text) {
  messageEl.textContent = text;
  messageEl.classList.remove("hidden");
  clearTimeout(messageTimer);
  messageTimer = setTimeout(
    () => messageEl.classList.add("hidden"), 2600);
}

async function refresh() {
  if (busy || aiRunning) return;
  const data = await api("/api/state");
  if (data) applyState(data);
}

function applyState(data) {
  const changed = JSON.stringify(data) !== JSON.stringify(state);
  state = data;
  if (changed) render();
  syncAiControls();
  maybeRunAiTurn();
}

let syncingAi = false;
function syncAiControls() {
  if (!state || syncingAi) return;
  syncingAi = true;
  if (state.ai) {
    aiAxisEl.checked = !!state.ai.axis;
    aiSovietEl.checked = !!state.ai.soviet;
  }
  if (state.ai_types) {
    const axisRadio = document.querySelector(`input[name="ai-type-axis"][value="${state.ai_types.axis || 'heuristic'}"]`);
    if (axisRadio) axisRadio.checked = true;
    const sovietRadio = document.querySelector(`input[name="ai-type-soviet"][value="${state.ai_types.soviet || 'heuristic'}"]`);
    if (sovietRadio) sovietRadio.checked = true;
  }
  if (state.training) {
    const t = state.training;
    const steps = t.checkpoint_steps || {};
    const axisCurrent = (state.ai_models && state.ai_models.axis) || (t.team_checkpoints && t.team_checkpoints.axis) || t.active_checkpoint;
    const sovietCurrent = (state.ai_models && state.ai_models.soviet) || (t.team_checkpoints && t.team_checkpoints.soviet) || t.active_checkpoint;
    populateSelect(modelSelectAxisEl, t.checkpoints, axisCurrent, "No checkpoint available", false, steps);
    populateSelect(modelSelectSovietEl, t.checkpoints, sovietCurrent, "No checkpoint available", false, steps);
    renderTraining(t);
  }
  if (state.ai_models) {
    if (state.ai_models.axis && modelSelectAxisEl.value !== state.ai_models.axis) {
      modelSelectAxisEl.value = state.ai_models.axis;
    }
    if (state.ai_models.soviet && modelSelectSovietEl.value !== state.ai_models.soviet) {
      modelSelectSovietEl.value = state.ai_models.soviet;
    }
  }
  syncingAi = false;
}

function populateSelect(selectEl, checkpoints, selectedVal, defaultText, allowEmpty, stepsMap) {
  if (!selectEl) return;
  const prevVal = selectEl.value;
  const targetVal = (selectedVal !== undefined && selectedVal !== null) ? selectedVal : prevVal;
  const validCheckpoints = (checkpoints || []).filter(Boolean);

  const existingOptions = Array.from(selectEl.options);
  const existingValues = existingOptions.map((o) => o.value);
  const expectedValues = [];
  if (allowEmpty) expectedValues.push("");
  if (validCheckpoints.length > 0) {
    expectedValues.push(...validCheckpoints);
  } else if (!allowEmpty) {
    expectedValues.push("");
  }

  const isSame = existingValues.length === expectedValues.length &&
    existingValues.every((val, idx) => val === expectedValues[idx]);

  if (isSame) {
    if (document.activeElement !== selectEl && targetVal !== undefined && targetVal !== null && selectEl.value !== targetVal) {
      selectEl.value = targetVal;
    }
    return;
  }

  selectEl.innerHTML = "";
  if (allowEmpty) {
    const emptyOpt = document.createElement("option");
    emptyOpt.value = "";
    emptyOpt.textContent = defaultText;
    if (targetVal === "") {
      emptyOpt.selected = true;
    }
    selectEl.appendChild(emptyOpt);
  }
  if (validCheckpoints.length > 0) {
    for (const cp of validCheckpoints) {
      const opt = document.createElement("option");
      opt.value = cp;
      const stepCount = stepsMap && stepsMap[cp] !== undefined ? stepsMap[cp] : null;
      if (stepCount !== null && stepCount !== undefined) {
        opt.textContent = `${cp} (${stepCount.toLocaleString()} steps)`;
      } else {
        opt.textContent = cp;
      }
      if (cp === targetVal) {
        opt.selected = true;
      }
      selectEl.appendChild(opt);
    }
  } else if (!allowEmpty) {
    selectEl.innerHTML = `<option value="">${defaultText}</option>`;
  }
  if (targetVal !== undefined && targetVal !== null) {
    selectEl.value = targetVal;
  }
}

window.api = api;
window.showMessage = showMessage;
window.populateSelect = populateSelect;

function renderTraining(t) {
  if (window.TrainingHub) {
    window.TrainingHub.render(t);
  }
}


async function maybeRunAiTurn() {
  if (!state || state.winner || aiRunning) return;
  if (!state.ai || !state.ai[state.turn]) return;
  aiRunning = true;
  const data = await api("/api/ai_turn", {});
  aiRunning = false;
  if (data) { clearSelection(); applyState(data); }
}

function unitAt(x, y) {
  return state.units.find((u) => u.x === x && u.y === y) || null;
}

function selectedUnit() {
  return state.units.find((u) => u.id === selectedId) || null;
}

function clearSelection() {
  selectedId = null;
  moves = [];
  targets = [];
}

async function select(unit) {
  selectedId = unit.id;
  moves = [];
  targets = [];
  render();
  const data = await api("/api/legal_moves", { unit_id: unit.id });
  if (data && selectedId === unit.id) {
    moves = data.moves || [];
    targets = data.targets || [];
    render();
  }
}

async function onCellClick(x, y) {
  if (busy || !state || state.winner) return;
  if (moves.some((m) => m[0] === x && m[1] === y)) {
    busy = true;
    const data = await api("/api/move", { unit_id: selectedId, x, y });
    busy = false;
    if (data) { clearSelection(); applyState(data); }
    return;
  }
  const clicked = unitAt(x, y);
  if (clicked && targets.includes(clicked.id)) {
    busy = true;
    const data = await api("/api/attack", {
      attacker_id: selectedId, target_id: clicked.id,
    });
    busy = false;
    if (data) { clearSelection(); applyState(data); }
    return;
  }
  if (clicked && clicked.team === state.turn) select(clicked);
  else clearSelection();
  render();
}

function render() {
  renderBanner();
  renderBoard();
  renderUnitPanel();
  renderObjectives();
  renderLog();
  renderOverlay();
}

function renderBanner() {
  bannerEl.textContent = state.winner
    ? "Battle over"
    : "Round " + state.round + " / " + state.max_rounds +
      " \u2014 " + state.turn_name + " to move";
  bannerEl.className = "banner " + state.turn;
}

function renderBoard() {
  boardEl.innerHTML = "";
  boardEl.style.gridTemplateColumns =
    "repeat(" + state.board.width + ", 46px)";
  for (let y = 0; y < state.board.height; y++) {
    for (let x = 0; x < state.board.width; x++) {
      boardEl.appendChild(makeCell(x, y));
    }
  }
}

function makeCell(x, y) {
  const cell = document.createElement("div");
  cell.className = "cell terrain-" + state.board.terrain[y][x];
  const obj = state.objectives.find((o) => o.x === x && o.y === y);
  if (obj) {
    cell.classList.add("objective");
    if (obj.controlled_by) cell.classList.add("held-" + obj.controlled_by);
    cell.title = obj.name;
  }
  if (moves.some((m) => m[0] === x && m[1] === y)) {
    cell.classList.add("hl-move");
  }
  const unit = unitAt(x, y);
  if (unit) {
    cell.classList.add("unit-" + unit.team);
    if (unit.id === selectedId) cell.classList.add("selected");
    if (targets.includes(unit.id)) cell.classList.add("hl-target");
    const token = document.createElement("span");
    token.className = "unit unit-" + unit.type + " unit-" + unit.team;
    token.textContent = UNIT_ICONS[unit.type] || "?";
    token.title = unit.name + " \u2014 " + unit.hp + "/" + unit.max_hp +
      " HP, atk " + unit.attack + ", move " + unit.move;
    cell.appendChild(token);
  }
  cell.addEventListener("click", () => onCellClick(x, y));
  return cell;
}

function renderUnitPanel() {
  const unit = selectedUnit();
  if (!unit) {
    unitDetailsEl.textContent = "Click one of your units.";
    return;
  }
  const status = unit.moved && unit.attacked
    ? "Done for this turn."
    : unit.attacked ? "Attacked." : unit.moved ? "Moved." : "Ready.";
  unitDetailsEl.innerHTML =
    "<b>" + unit.name + "</b> (" + unit.team + ")<br>" +
    "HP: " + unit.hp + "/" + unit.max_hp +
    " &nbsp; Attack: " + unit.attack +
    " &nbsp; Range: " + unit.range +
    " &nbsp; Move: " + unit.move + "<br>" + status;
}

function renderObjectives() {
  objectiveListEl.innerHTML = "";
  state.objectives.forEach((obj) => {
    const li = document.createElement("li");
    const holder = obj.controlled_by
      ? "<span class='ctrl-" + obj.controlled_by + "'>" +
        (obj.controlled_by === "axis" ? "Axis" : "Soviets") + "</span>"
      : "<span class='ctrl-none'>unclaimed</span>";
    li.innerHTML = obj.name + " (" + obj.x + "," + obj.y + ") \u2014 " + holder;
    objectiveListEl.appendChild(li);
  });
}

function renderLog() {
  logListEl.innerHTML = "";
  state.log.forEach((entry) => {
    const li = document.createElement("li");
    li.textContent = entry;
    logListEl.appendChild(li);
  });
  logListEl.scrollTop = logListEl.scrollHeight;
}

let overlayDismissed = false;

function renderOverlay() {
  if (!state.winner || overlayDismissed) {
    overlayEl.classList.add("hidden");
    return;
  }
  overlayEl.classList.remove("hidden");
  const reason = state.log[state.log.length - 1] || "";
  if (state.winner === "axis") {
    overlayTitleEl.textContent = "Axis victory!";
    overlayTitleEl.style.color = "#2b3948";
  } else {
    overlayTitleEl.textContent = "The Soviets hold Stalingrad!";
    overlayTitleEl.style.color = "#a02a22";
  }
  overlayTextEl.textContent = reason;
}

function dismissOverlay() {
  overlayDismissed = true;
  overlayEl.classList.add("hidden");
}

document.getElementById("btn-overlay-close").addEventListener("click", dismissOverlay);
document.getElementById("btn-close-overlay").addEventListener("click", dismissOverlay);

overlayEl.addEventListener("click", (e) => {
  if (e.target === overlayEl) dismissOverlay();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !overlayEl.classList.contains("hidden")) {
    dismissOverlay();
  }
});

document.getElementById("btn-end-turn").addEventListener("click", async () => {
  if (busy || !state || state.winner) return;
  busy = true;
  const data = await api("/api/end_turn", {});
  busy = false;
  if (data) { clearSelection(); applyState(data); }
});

async function resetGame() {
  overlayDismissed = false;
  busy = true;
  const data = await api("/api/reset", {});
  busy = false;
  if (data) { clearSelection(); applyState(data); }
}

document.getElementById("btn-reset").addEventListener("click", () => {
  if (confirm("Restart the battle from the beginning?")) resetGame();
});

document.getElementById("btn-again").addEventListener("click", resetGame);

async function onAiToggle() {
  if (syncingAi || !state) return;
  busy = true;
  const data = await api("/api/set_ai", {
    axis: aiAxisEl.checked,
    soviet: aiSovietEl.checked,
  });
  busy = false;
  if (data) { clearSelection(); applyState(data); }
}

aiAxisEl.addEventListener("change", onAiToggle);
aiSovietEl.addEventListener("change", onAiToggle);

modelSelectAxisEl.addEventListener("change", async () => {
  const selected = modelSelectAxisEl.value;
  if (!selected) return;
  const data = await api("/api/set_ai_type", { axis_model: selected });
  if (data) applyState(data);
});

modelSelectSovietEl.addEventListener("change", async () => {
  const selected = modelSelectSovietEl.value;
  if (!selected) return;
  const data = await api("/api/set_ai_type", { soviet_model: selected });
  if (data) applyState(data);
});

document.querySelectorAll('input[name="ai-type-axis"], input[name="ai-type-soviet"]').forEach((radio) => {
  radio.addEventListener("change", async () => {
    const axisEl = document.querySelector('input[name="ai-type-axis"]:checked');
    const sovietEl = document.querySelector('input[name="ai-type-soviet"]:checked');
    if (!axisEl || !sovietEl) return;
    const data = await api("/api/set_ai_type", {
      axis: axisEl.value,
      soviet: sovietEl.value,
      axis_model: modelSelectAxisEl.value || undefined,
      soviet_model: modelSelectSovietEl.value || undefined,
    });
    if (data) applyState(data);
  });
});

setInterval(refresh, 2000);
refresh();

