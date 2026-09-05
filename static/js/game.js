"use strict";

/* Stalingrad 1942 - frontend. Renders server state, sends actions to API. */

let state = null;
let selectedId = null;
let moves = [];
let targets = [];
let busy = false;

const boardEl = document.getElementById("board");
const bannerEl = document.getElementById("banner");
const unitDetailsEl = document.getElementById("unit-details");
const objectiveListEl = document.getElementById("objective-list");
const logListEl = document.getElementById("log-list");
const messageEl = document.getElementById("message");
const overlayEl = document.getElementById("overlay");
const overlayTitleEl = document.getElementById("overlay-title");
const overlayTextEl = document.getElementById("overlay-text");

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
  if (busy) return;
  const data = await api("/api/state");
  if (data) applyState(data);
}

function applyState(data) {
  const changed = JSON.stringify(data) !== JSON.stringify(state);
  state = data;
  if (changed) render();
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

function renderOverlay() {
  if (!state.winner) {
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

document.getElementById("btn-end-turn").addEventListener("click", async () => {
  if (busy || !state || state.winner) return;
  busy = true;
  const data = await api("/api/end_turn", {});
  busy = false;
  if (data) { clearSelection(); applyState(data); }
});

async function resetGame() {
  busy = true;
  const data = await api("/api/reset", {});
  busy = false;
  if (data) { clearSelection(); applyState(data); }
}

document.getElementById("btn-reset").addEventListener("click", () => {
  if (confirm("Restart the battle from the beginning?")) resetGame();
});

document.getElementById("btn-again").addEventListener("click", resetGame);

setInterval(refresh, 2500);
refresh();
