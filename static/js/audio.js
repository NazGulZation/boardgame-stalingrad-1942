"use strict";

/* Stalingrad 1942 - Audio Manager.
   Handles unit-specific movement and combat sound effects,
   volume/mute settings, and log-based AI action audio cues. */

const SoundManager = (() => {
  const STORAGE_KEY_MUTED = "stalingrad_sfx_muted";
  const STORAGE_KEY_VOLUME = "stalingrad_sfx_volume";

  const SOUND_PATHS = {
    rifle_move: "/static/audio/rifle_move.mp3",
    rifle_attack: "/static/audio/rifle_attack.mp3",
    sniper_move: "/static/audio/sniper_move.mp3",
    sniper_attack: "/static/audio/sniper_attack.mp3",
    tank_move: "/static/audio/tank_move.mp3",
    tank_attack: "/static/audio/tank_attack.mp3",
    unit_destroyed: "/static/audio/unit_destroyed.mp3",
    unit_select: "/static/audio/unit_select.mp3",
    victory: "/static/audio/victory.mp3",
  };

  let isMuted = localStorage.getItem(STORAGE_KEY_MUTED) === "true";
  let volume = 0.75;
  const storedVol = localStorage.getItem(STORAGE_KEY_VOLUME);
  if (storedVol !== null && !isNaN(parseFloat(storedVol))) {
    volume = Math.max(0, Math.min(1, parseFloat(storedVol)));
  }

  // Audio elements cache
  const audioCache = {};
  let unlocked = false;

  function initAudio() {
    if (unlocked) return;
    unlocked = true;
    for (const [key, path] of Object.entries(SOUND_PATHS)) {
      try {
        const audio = new Audio(path);
        audio.preload = "auto";
        audioCache[key] = audio;
      } catch (err) {
        // Audio not supported or blocked
      }
    }
  }

  // Unlock audio on first user interaction
  const unlockEvents = ["click", "keydown", "touchstart", "pointerdown"];
  function onFirstUserGesture() {
    initAudio();
    unlockEvents.forEach((ev) => window.removeEventListener(ev, onFirstUserGesture));
  }
  unlockEvents.forEach((ev) => window.addEventListener(ev, onFirstUserGesture, { once: true }));

  function play(soundName, relativeVolume = 1.0) {
    if (isMuted || volume <= 0) return;
    initAudio();
    const path = SOUND_PATHS[soundName];
    if (!path) return;

    try {
      // Create clone for polyphonic overlap
      const sound = new Audio(path);
      sound.volume = Math.max(0, Math.min(1, volume * relativeVolume));
      const playPromise = sound.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {
          // Playback blocked or interrupted
        });
      }
    } catch (err) {
      // Audio playback failed gracefully
    }
  }

  function playMove(unitType) {
    const key = `${unitType}_move`;
    if (SOUND_PATHS[key]) {
      play(key, 0.9);
    }
  }

  function playAttack(unitType, isDestroyed = false) {
    const key = `${unitType}_attack`;
    if (SOUND_PATHS[key]) {
      play(key, 1.0);
    }
    if (isDestroyed) {
      setTimeout(() => {
        play("unit_destroyed", 0.95);
      }, 500);
    }
  }

  function playSelect() {
    play("unit_select", 0.4);
  }

  function playVictory() {
    play("victory", 0.85);
  }

  function playDestroyed() {
    play("unit_destroyed", 0.95);
  }

  function setMuted(muted) {
    isMuted = !!muted;
    localStorage.setItem(STORAGE_KEY_MUTED, isMuted ? "true" : "false");
    updateUi();
  }

  function toggleMute() {
    setMuted(!isMuted);
    return isMuted;
  }

  function setVolume(newVol) {
    volume = Math.max(0, Math.min(1, parseFloat(newVol) || 0));
    localStorage.setItem(STORAGE_KEY_VOLUME, volume.toFixed(2));
    if (volume === 0) {
      setMuted(true);
    } else if (isMuted) {
      setMuted(false);
    }
    updateUi();
  }

  function getSettings() {
    return { isMuted, volume };
  }

  function updateUi() {
    const btnMute = document.getElementById("btn-sound-toggle");
    const slider = document.getElementById("sound-volume");
    if (btnMute) {
      btnMute.textContent = isMuted || volume === 0 ? "🔇" : "🔊";
      btnMute.setAttribute("aria-label", isMuted ? "Unmute Sound" : "Mute Sound");
      btnMute.classList.toggle("muted", isMuted || volume === 0);
    }
    if (slider) {
      slider.value = isMuted ? 0 : Math.round(volume * 100);
    }
  }

  /**
   * Parses new battle log lines to play unit sounds during AI turns.
   */
  function playLogEvents(newEntries) {
    if (!newEntries || !newEntries.length || isMuted) return;
    let delay = 100;

    for (const entry of newEntries) {
      const lower = entry.toLowerCase();
      let unitType = null;
      if (lower.includes("rifle squad")) unitType = "rifle";
      else if (lower.includes("sniper team")) unitType = "sniper";
      else if (lower.includes("tank")) unitType = "tank";

      if (lower.includes("moved to") && unitType) {
        const u = unitType;
        setTimeout(() => playMove(u), delay);
        delay += 600;
      } else if ((lower.includes("hit") || lower.includes("destroyed")) && unitType) {
        const u = unitType;
        const isDestroyed = lower.includes("destroyed");
        setTimeout(() => playAttack(u, isDestroyed), delay);
        delay += 750;
      }
    }
  }

  return {
    init: initAudio,
    play,
    playMove,
    playAttack,
    playSelect,
    playVictory,
    playDestroyed,
    playLogEvents,
    setMuted,
    toggleMute,
    setVolume,
    getSettings,
    updateUi,
  };
})();

window.SoundManager = SoundManager;
