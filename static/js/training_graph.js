/**
 * TrainingGraph - Real-time RL Reward Trend Graph for Stalingrad 1942
 * Features:
 * - HTML5 Canvas with High-DPI scaling
 * - Customizable step window (default 5000 steps)
 * - 5-second auto-update polling loop
 * - Hover crosshair: vertical line on hovered step with glowing point marker
 * - Floating tooltip and live step/reward readout
 */

(function () {
  const canvas = document.getElementById("reward-canvas");
  const windowSelect = document.getElementById("reward-graph-window");
  const tooltip = document.getElementById("reward-tooltip");
  const readout = document.getElementById("reward-hover-readout");

  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let history = []; // Array of [step, reward]
  let hoverPos = null; // { x, y }

  function getWindowSize() {
    if (!windowSelect) return 5000;
    const val = windowSelect.value;
    return val === "all" ? "all" : parseInt(val, 10) || 5000;
  }

  function getFilteredPoints() {
    if (!history || history.length === 0) return [];
    const w = getWindowSize();
    if (w === "all") return history.map(p => ({ step: p[0], reward: p[1] }));
    const maxStep = history[history.length - 1][0];
    const minStep = Math.max(0, maxStep - w);
    return history
      .filter(p => p[0] >= minStep)
      .map(p => ({ step: p[0], reward: p[1] }));
  }

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const width = Math.floor(rect.width) || 280;
    const height = Math.floor(rect.height) || 130;

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width, height };
  }

  function render() {
    const { width, height } = resizeCanvas();
    ctx.clearRect(0, 0, width, height);

    const pad = { top: 12, right: 12, bottom: 22, left: 40 };
    const plotW = Math.max(10, width - pad.left - pad.right);
    const plotH = Math.max(10, height - pad.top - pad.bottom);

    const points = getFilteredPoints();

    // Empty state
    if (points.length < 2) {
      ctx.fillStyle = "#475569";
      ctx.font = "11px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      const msg = points.length === 1
        ? `Step ${points[0].step}: Reward ${points[0].reward.toFixed(4)} (gathering...)`
        : "Start training to view live reward trend";
      ctx.fillText(msg, width / 2, height / 2);
      if (tooltip) tooltip.classList.add("hidden");
      if (readout && points.length === 1) {
        readout.textContent = `Step ${points[0].step} | Reward: ${points[0].reward.toFixed(4)}`;
      }
      return;
    }

    const minStep = points[0].step;
    const maxStep = points[points.length - 1].step;
    const stepSpan = Math.max(1, maxStep - minStep);

    let minR = Infinity;
    let maxR = -Infinity;
    for (const p of points) {
      if (p.reward < minR) minR = p.reward;
      if (p.reward > maxR) maxR = p.reward;
    }

    // Ensure a reasonable vertical range
    if (minR === maxR) {
      minR -= 0.1;
      maxR += 0.1;
    } else {
      const margin = (maxR - minR) * 0.12;
      minR -= margin;
      maxR += margin;
    }
    const rSpan = maxR - minR;

    const toX = (step) => pad.left + ((step - minStep) / stepSpan) * plotW;
    const toY = (r) => pad.top + plotH - ((r - minR) / rSpan) * plotH;

    // Draw horizontal grid lines & labels
    ctx.font = "9px monospace";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const numGridLines = 4;
    for (let i = 0; i <= numGridLines; i++) {
      const rVal = minR + (i / numGridLines) * rSpan;
      const y = toY(rVal);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();

      ctx.fillStyle = "#64748b";
      ctx.fillText((rVal >= 0 ? "+" : "") + rVal.toFixed(2), pad.left - 4, y);
    }

    // Zero reference line if within range
    if (minR < 0 && maxR > 0) {
      const zeroY = toY(0);
      ctx.save();
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "rgba(148, 163, 184, 0.4)";
      ctx.beginPath();
      ctx.moveTo(pad.left, zeroY);
      ctx.lineTo(pad.left + plotW, zeroY);
      ctx.stroke();
      ctx.restore();
    }

    // Bottom step labels
    ctx.fillStyle = "#64748b";
    ctx.textAlign = "left";
    ctx.fillText(`${minStep.toLocaleString()}`, pad.left, height - 6);
    ctx.textAlign = "right";
    ctx.fillText(`${maxStep.toLocaleString()}`, pad.left + plotW, height - 6);

    // Gradient fill under curve
    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + plotH);
    grad.addColorStop(0, "rgba(56, 189, 248, 0.3)");
    grad.addColorStop(1, "rgba(56, 189, 248, 0.0)");

    ctx.beginPath();
    ctx.moveTo(toX(points[0].step), toY(points[0].reward));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(toX(points[i].step), toY(points[i].reward));
    }
    ctx.lineTo(toX(points[points.length - 1].step), pad.top + plotH);
    ctx.lineTo(toX(points[0].step), pad.top + plotH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Stroke curve
    ctx.beginPath();
    ctx.moveTo(toX(points[0].step), toY(points[0].reward));
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(toX(points[i].step), toY(points[i].reward));
    }
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 1.8;
    ctx.lineJoin = "round";
    ctx.stroke();

    // Hover logic: vertical line & tooltip
    let activePoint = null;
    if (hoverPos && hoverPos.x >= pad.left && hoverPos.x <= pad.left + plotW) {
      // Find closest point by x coordinate
      let closestDist = Infinity;
      for (const p of points) {
        const px = toX(p.step);
        const dist = Math.abs(px - hoverPos.x);
        if (dist < closestDist) {
          closestDist = dist;
          activePoint = p;
        }
      }
    }

    if (activePoint) {
      const hx = toX(activePoint.step);
      const hy = toY(activePoint.reward);

      // Draw vertical crosshair line across the plot area
      ctx.save();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.75)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hx, pad.top);
      ctx.lineTo(hx, pad.top + plotH);
      ctx.stroke();
      ctx.restore();

      // Outer glow marker
      ctx.beginPath();
      ctx.arc(hx, hy, 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
      ctx.fill();

      // Core point marker
      ctx.beginPath();
      ctx.arc(hx, hy, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#38bdf8";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Update tooltip
      const sign = activePoint.reward >= 0 ? "+" : "";
      const text = `Step ${activePoint.step.toLocaleString()}: ${sign}${activePoint.reward.toFixed(4)}`;
      if (tooltip) {
        tooltip.textContent = text;
        tooltip.classList.remove("hidden");
        const ttWidth = tooltip.offsetWidth || 110;
        let ttLeft = hx - ttWidth / 2;
        ttLeft = Math.max(pad.left, Math.min(width - pad.right - ttWidth, ttLeft));
        let ttTop = hy - 26;
        if (ttTop < pad.top) ttTop = hy + 8;
        tooltip.style.left = `${ttLeft}px`;
        tooltip.style.top = `${ttTop}px`;
      }

      if (readout) {
        readout.textContent = `Step ${activePoint.step.toLocaleString()} | Reward: ${sign}${activePoint.reward.toFixed(4)}`;
      }
    } else {
      if (tooltip) tooltip.classList.add("hidden");
      if (readout) {
        const latest = points[points.length - 1];
        const sign = latest.reward >= 0 ? "+" : "";
        readout.textContent = `Latest: Step ${latest.step.toLocaleString()} | Reward: ${sign}${latest.reward.toFixed(4)}`;
      }
    }
  }

  function handlePointer(clientX, clientY) {
    const rect = canvas.getBoundingClientRect();
    hoverPos = {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
    render();
  }

  canvas.addEventListener("mousemove", (e) => {
    handlePointer(e.clientX, e.clientY);
  });

  canvas.addEventListener("mouseleave", () => {
    hoverPos = null;
    render();
  });

  canvas.addEventListener("touchstart", (e) => {
    if (e.touches && e.touches[0]) {
      handlePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  canvas.addEventListener("touchmove", (e) => {
    if (e.touches && e.touches[0]) {
      handlePointer(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  canvas.addEventListener("touchend", () => {
    hoverPos = null;
    render();
  });

  if (windowSelect) {
    windowSelect.addEventListener("change", () => {
      render();
    });
  }

  window.addEventListener("resize", () => {
    render();
  });

  function setData(newHistory) {
    if (Array.isArray(newHistory)) {
      history = newHistory;
      render();
    }
  }

  // 5-second automatic update polling loop
  async function pollTrainingStatus() {
    try {
      const res = await fetch("/api/training/status");
      if (res.ok) {
        const data = await res.json();
        if (data && Array.isArray(data.reward_history)) {
          history = data.reward_history;
          render();
        }
      }
    } catch (_) {
      // Ignore network errors during polling
    }
  }

  setInterval(pollTrainingStatus, 5000);

  // Initial draw
  render();

  // Export to window for game.js integration
  window.TrainingGraph = {
    setData,
    render,
  };
})();
