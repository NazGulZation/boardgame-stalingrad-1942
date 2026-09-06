"use strict";

/**
 * TrainingHub - Manages RL Training UI, Opponent Pool Checklist, and Subprocess Status
 */
(function () {
  const trainSideSelectEl = document.getElementById("train-side-select");
  const oppHeuristicCheckEl = document.getElementById("opp-check-heuristic");
  const oppHeuristicLabelEl = document.getElementById("opp-heuristic-label");
  const oppCheckpointsListEl = document.getElementById("opp-checkpoints-list");
  const oppPoolCountEl = document.getElementById("opp-pool-count");
  const btnOppAllEl = document.getElementById("btn-opp-all");
  const btnOppHeuristicEl = document.getElementById("btn-opp-heuristic");
  const trainPoolBreakdownEl = document.getElementById("train-pool-breakdown");
  const trainResumeSelectEl = document.getElementById("train-resume-select");
  const trainResumeInfoEl = document.getElementById("train-resume-info");
  const trainResumeStepsEl = document.getElementById("train-resume-steps");
  const trainStatusBadgeEl = document.getElementById("train-status-badge");
  const trainStepsEl = document.getElementById("train-steps");
  const trainStepsCustomEl = document.getElementById("train-steps-custom");
  const wrapCustomStepsEl = document.getElementById("wrap-custom-steps");
  const trainModelNameEl = document.getElementById("train-model-name");
  const trainParallelSelectEl = document.getElementById("train-parallel-select");
  const btnTrainStartEl = document.getElementById("btn-train-start");
  const btnTrainStopEl = document.getElementById("btn-train-stop");
  const trainProgressBarEl = document.getElementById("train-progress-bar");
  const trainProgressTextEl = document.getElementById("train-progress-text");
  const trainSpsTextEl = document.getElementById("train-sps-text");
  const trainWinRateEl = document.getElementById("train-win-rate");
  const trainWinRateLabelEl = document.getElementById("train-win-rate-label");
  const trainRewardEl = document.getElementById("train-reward");
  const trainPolicyLossEl = document.getElementById("train-policy-loss");
  const trainValueLossEl = document.getElementById("train-value-loss");
  const trainElapsedEl = document.getElementById("train-elapsed");

  let checkpointSteps = {};
  let selectedOpponents = new Set(["heuristic"]);
  let prevTrainStatus = "idle";

  function updateModelNamePlaceholder() {
    if (!trainModelNameEl) return;
    const side = trainSideSelectEl ? trainSideSelectEl.value : "axis";
    const isResume = trainResumeSelectEl && !!trainResumeSelectEl.value;
    if (isResume) {
      trainModelNameEl.placeholder = "e.g. overwrite or new name";
    } else {
      const prefix = `stalingrad_${side}_v`;
      trainModelNameEl.placeholder = `e.g. ${prefix} (optional)`;
    }
  }

  function updateResumeStepInfo() {
    if (!trainResumeInfoEl || !trainResumeStepsEl || !trainResumeSelectEl) return;
    const val = trainResumeSelectEl.value;
    if (!val) {
      trainResumeInfoEl.className = "resume-info-box scratch";
      trainResumeStepsEl.textContent = "Fresh start (0 prior steps)";
    } else {
      trainResumeInfoEl.className = "resume-info-box";
      const steps = checkpointSteps[val];
      if (steps !== undefined && steps !== null) {
        trainResumeStepsEl.textContent = `${steps.toLocaleString()} steps done`;
      } else {
        trainResumeStepsEl.textContent = "Unknown steps";
      }
    }
  }

  function updateOpponentHeuristicLabel() {
    if (!oppHeuristicLabelEl || !trainSideSelectEl) return;
    if (trainSideSelectEl.value === "axis") {
      oppHeuristicLabelEl.textContent = "Bot Heuristic (Soviet 62nd Army)";
    } else {
      oppHeuristicLabelEl.textContent = "Bot Heuristic (Axis 6th Army)";
    }
  }

  function updateOpponentCountBadge() {
    if (!oppPoolCountEl) return;
    const count = (oppHeuristicCheckEl && oppHeuristicCheckEl.checked ? 1 : 0) +
      (oppCheckpointsListEl ? oppCheckpointsListEl.querySelectorAll("input[type='checkbox']:checked").length : 0);
    oppPoolCountEl.textContent = `${count} selected`;
    if (count === 0) {
      oppPoolCountEl.classList.add("warn");
    } else {
      oppPoolCountEl.classList.remove("warn");
    }
  }

  function renderOpponentChecklist(checkpoints, isRunning) {
    if (!oppCheckpointsListEl) return;
    const validCheckpoints = (checkpoints || []).filter(Boolean);

    if (validCheckpoints.length === 0) {
      oppCheckpointsListEl.innerHTML = '<div class="opp-empty-hint">No model checkpoints yet. Train vs Heuristic first.</div>';
      updateOpponentCountBadge();
      return;
    }

    const currentRendered = oppCheckpointsListEl.querySelectorAll("input[type='checkbox']");
    if (currentRendered.length > 0) {
      currentRendered.forEach((cb) => {
        if (cb.checked) selectedOpponents.add(cb.value);
        else selectedOpponents.delete(cb.value);
      });
    }

    oppCheckpointsListEl.innerHTML = "";
    validCheckpoints.forEach((cp) => {
      const label = document.createElement("label");
      label.className = "opp-item";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = cp;
      cb.disabled = isRunning;
      cb.checked = selectedOpponents.has(cp);

      cb.addEventListener("change", () => {
        if (cb.checked) selectedOpponents.add(cp);
        else selectedOpponents.delete(cp);
        updateOpponentCountBadge();
      });

      const nameSpan = document.createElement("span");
      nameSpan.className = "opp-label";
      nameSpan.textContent = cp;
      nameSpan.title = cp;

      const badgeSpan = document.createElement("span");
      badgeSpan.className = "opp-badge badge-model";
      const steps = checkpointSteps[cp];
      if (steps !== undefined && steps !== null) {
        badgeSpan.textContent = `${(steps / 1000).toFixed(0)}k steps`;
      } else {
        badgeSpan.textContent = "RL Model";
      }

      label.appendChild(cb);
      label.appendChild(nameSpan);
      label.appendChild(badgeSpan);
      oppCheckpointsListEl.appendChild(label);
    });

    updateOpponentCountBadge();
  }

  function renderTraining(t) {
    if (!t) return;
    const statusStr = t.status || "idle";
    if (trainStatusBadgeEl) {
      trainStatusBadgeEl.textContent = statusStr.toUpperCase();
      trainStatusBadgeEl.className = "badge " + statusStr.toLowerCase();
    }

    const isRunning = !!t.is_running || statusStr === "training";
    if (btnTrainStartEl) btnTrainStartEl.disabled = isRunning;
    if (btnTrainStopEl) btnTrainStopEl.disabled = !isRunning;
    if (trainSideSelectEl) trainSideSelectEl.disabled = isRunning;
    if (oppHeuristicCheckEl) oppHeuristicCheckEl.disabled = isRunning;
    if (btnOppAllEl) btnOppAllEl.disabled = isRunning;
    if (btnOppHeuristicEl) btnOppHeuristicEl.disabled = isRunning;
    if (trainStepsEl) trainStepsEl.disabled = isRunning;
    if (trainStepsCustomEl) trainStepsCustomEl.disabled = isRunning;
    if (trainResumeSelectEl) trainResumeSelectEl.disabled = isRunning;
    if (trainModelNameEl) trainModelNameEl.disabled = isRunning;
    if (trainParallelSelectEl) trainParallelSelectEl.disabled = isRunning;

    updateOpponentHeuristicLabel();
    renderOpponentChecklist(t.checkpoints, isRunning);

    const pct = Math.min(100, Math.max(0, t.progress || 0));
    if (trainProgressBarEl) trainProgressBarEl.style.width = pct + "%";
    if (trainProgressTextEl) {
      trainProgressTextEl.textContent = `${pct}% (${(t.step || 0).toLocaleString()} / ${(t.total_steps || 0).toLocaleString()})`;
    }
    if (trainSpsTextEl) trainSpsTextEl.textContent = `${t.sps || 0} SPS`;

    if (trainWinRateLabelEl) {
      if (t.opponents && t.opponents.length > 1) {
        trainWinRateLabelEl.textContent = `Win Rate (${t.opponents.length} Opponents Pool Avg)`;
        trainWinRateLabelEl.title = `Evaluated across ${t.opponents.length} opponents in pool`;
      } else if (t.opponent === "checkpoint" || t.opponent_checkpoint) {
        trainWinRateLabelEl.textContent = "Win vs Model";
        trainWinRateLabelEl.title = t.opponent_checkpoint ? `Evaluated against ${t.opponent_checkpoint}` : "Evaluated against checkpoint model";
      } else {
        trainWinRateLabelEl.textContent = "Win vs AI";
        trainWinRateLabelEl.title = "Evaluated against rule-based Heuristic AI";
      }
    }

    if (trainPoolBreakdownEl) {
      if (t.pool_eval && Object.keys(t.pool_eval).length > 1) {
        const parts = Object.entries(t.pool_eval).map(([name, wr]) => `${name}: ${wr}%`);
        trainPoolBreakdownEl.textContent = parts.join(" • ");
        trainPoolBreakdownEl.classList.remove("hidden");
      } else {
        trainPoolBreakdownEl.textContent = "";
        trainPoolBreakdownEl.classList.add("hidden");
      }
    }

    if (trainWinRateEl) {
      trainWinRateEl.textContent = t.win_rate !== undefined && t.win_rate !== null ? `${t.win_rate}%` : "--%";
    }
    if (trainRewardEl) {
      trainRewardEl.textContent = (t.reward !== undefined && t.reward !== null)
        ? ((t.reward >= 0 ? "+" : "") + Number(t.reward).toFixed(3))
        : "--";
    }
    if (trainPolicyLossEl) {
      trainPolicyLossEl.textContent = t.policy_loss !== undefined && t.policy_loss !== null ? t.policy_loss : "--";
    }
    if (trainValueLossEl) {
      trainValueLossEl.textContent = t.value_loss !== undefined && t.value_loss !== null ? t.value_loss : "--";
    }
    if (trainElapsedEl) {
      trainElapsedEl.textContent = `${Math.round(t.elapsed || 0)}s`;
    }

    if (window.TrainingGraph && Array.isArray(t.reward_history)) {
      window.TrainingGraph.setData(t.reward_history);
    }

    if (t.checkpoint_steps) {
      checkpointSteps = Object.assign({}, checkpointSteps, t.checkpoint_steps);
    }

    if (trainResumeSelectEl && window.populateSelect) {
      window.populateSelect(trainResumeSelectEl, t.checkpoints, trainResumeSelectEl.value, "Start from scratch (New model)", true, checkpointSteps);
    }
    updateResumeStepInfo();
    updateModelNamePlaceholder();

    if (prevTrainStatus === "training" && statusStr === "completed") {
      const cpName = t.latest_checkpoint ? t.latest_checkpoint.split(/[\\/]/).pop() : (t.model_name || "checkpoint");
      if (window.showMessage) window.showMessage(`Training completed! New model saved: ${cpName}`);
      if (trainResumeSelectEl && cpName) {
        trainResumeSelectEl.value = cpName;
        updateResumeStepInfo();
        updateModelNamePlaceholder();
      }
    }
    prevTrainStatus = statusStr;
  }

  // Setup Event Listeners
  if (btnOppAllEl) {
    btnOppAllEl.addEventListener("click", () => {
      if (btnTrainStartEl && btnTrainStartEl.disabled) return;
      if (oppHeuristicCheckEl) oppHeuristicCheckEl.checked = true;
      selectedOpponents.add("heuristic");
      if (oppCheckpointsListEl) {
        oppCheckpointsListEl.querySelectorAll("input[type='checkbox']").forEach((cb) => {
          cb.checked = true;
          selectedOpponents.add(cb.value);
        });
      }
      updateOpponentCountBadge();
    });
  }

  if (btnOppHeuristicEl) {
    btnOppHeuristicEl.addEventListener("click", () => {
      if (btnTrainStartEl && btnTrainStartEl.disabled) return;
      if (oppHeuristicCheckEl) oppHeuristicCheckEl.checked = true;
      selectedOpponents = new Set(["heuristic"]);
      if (oppCheckpointsListEl) {
        oppCheckpointsListEl.querySelectorAll("input[type='checkbox']").forEach((cb) => {
          cb.checked = false;
        });
      }
      updateOpponentCountBadge();
    });
  }

  if (oppHeuristicCheckEl) {
    oppHeuristicCheckEl.addEventListener("change", () => {
      if (oppHeuristicCheckEl.checked) selectedOpponents.add("heuristic");
      else selectedOpponents.delete("heuristic");
      updateOpponentCountBadge();
    });
  }

  if (trainSideSelectEl) {
    trainSideSelectEl.addEventListener("change", () => {
      updateModelNamePlaceholder();
      updateOpponentHeuristicLabel();
    });
  }

  if (trainResumeSelectEl) {
    trainResumeSelectEl.addEventListener("change", () => {
      updateResumeStepInfo();
      updateModelNamePlaceholder();
    });
  }

  if (trainStepsEl && wrapCustomStepsEl) {
    trainStepsEl.addEventListener("change", () => {
      if (trainStepsEl.value === "custom") {
        wrapCustomStepsEl.classList.remove("hidden");
        if (trainStepsCustomEl) {
          if (!trainStepsCustomEl.value) trainStepsCustomEl.value = "20000";
          trainStepsCustomEl.focus();
          trainStepsCustomEl.select();
        }
      } else {
        wrapCustomStepsEl.classList.add("hidden");
      }
    });
  }

  if (btnTrainStartEl) {
    btnTrainStartEl.addEventListener("click", async () => {
      let steps;
      if (trainStepsEl && trainStepsEl.value === "custom") {
        steps = parseInt(trainStepsCustomEl ? trainStepsCustomEl.value : "", 10);
        if (isNaN(steps) || steps <= 0) {
          if (window.showMessage) window.showMessage("Please enter a valid positive number of timesteps.");
          if (trainStepsCustomEl) trainStepsCustomEl.focus();
          return;
        }
        if (steps < 100) {
          if (window.showMessage) window.showMessage("Timesteps must be at least 100.");
          if (trainStepsCustomEl) trainStepsCustomEl.focus();
          return;
        }
      } else {
        steps = parseInt(trainStepsEl ? trainStepsEl.value : "20000", 10) || 20000;
      }

      const trainSide = trainSideSelectEl ? trainSideSelectEl.value : "axis";

      const checkedOpponents = [];
      if (oppHeuristicCheckEl && oppHeuristicCheckEl.checked) {
        checkedOpponents.push("heuristic");
      }
      if (oppCheckpointsListEl) {
        oppCheckpointsListEl.querySelectorAll("input[type='checkbox']:checked").forEach((cb) => {
          checkedOpponents.push(cb.value);
        });
      }

      if (checkedOpponents.length === 0) {
        if (window.showMessage) window.showMessage("Pilih minimal satu lawan untuk training.");
        if (oppPoolCountEl) oppPoolCountEl.classList.add("warn");
        btnTrainStartEl.disabled = false;
        return;
      }

      const resumeCp = trainResumeSelectEl ? trainResumeSelectEl.value || null : null;
      const customModelName = trainModelNameEl ? trainModelNameEl.value.trim() : "";
      const nameSuffix = customModelName ? ` as "${customModelName}"` : "";
      const parallelEnvs = parseInt(trainParallelSelectEl ? trainParallelSelectEl.value : "4", 10) || 4;
      btnTrainStartEl.disabled = true;
      const sideLabel = trainSide === "axis" ? "Axis (Nazi)" : "Soviet";
      const oppLabel = checkedOpponents.length > 1
        ? `Pool (${checkedOpponents.length} opponents round-robin)`
        : (checkedOpponents[0] === "heuristic" ? "Heuristic AI" : checkedOpponents[0]);

      if (resumeCp) {
        const priorSteps = checkpointSteps[resumeCp];
        const stepText = (priorSteps !== undefined && priorSteps !== null) ? ` (${priorSteps.toLocaleString()} steps already done)` : "";
        if (window.showMessage) window.showMessage(`Resuming ${sideLabel} training vs ${oppLabel} from ${resumeCp}${stepText}${nameSuffix} [${parallelEnvs}x parallel] for ${steps.toLocaleString()} steps on GPU...`);
      } else {
        if (window.showMessage) window.showMessage(`Launching ${sideLabel} training vs ${oppLabel} (${steps.toLocaleString()} steps)${nameSuffix} [${parallelEnvs}x parallel] from scratch on GPU...`);
      }

      if (window.api) {
        const res = await window.api("/api/training/start", {
          total_timesteps: steps,
          num_envs: parallelEnvs,
          resume_checkpoint: resumeCp,
          train_side: trainSide,
          opponents: checkedOpponents,
          model_name: customModelName || null,
        });
        if (res && res.training) renderTraining(res.training);
      }
    });
  }

  if (btnTrainStopEl) {
    btnTrainStopEl.addEventListener("click", async () => {
      btnTrainStopEl.disabled = true;
      if (window.showMessage) window.showMessage("Stopping RL training run...");
      if (window.api) {
        const res = await window.api("/api/training/stop", {});
        if (res && res.training) renderTraining(res.training);
      }
    });
  }

  window.TrainingHub = {
    render: renderTraining,
    getCheckpointSteps: () => checkpointSteps,
  };
})();
