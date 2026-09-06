"""Training Manager for Stalingrad 1942 RL.

Manages background CleanRL PPO training subprocess, real-time status reporting,
checkpoint scanning, and cached RLAgent inference for the web app.
"""

import os
import sys
import json
import subprocess
import time

CONDA_RL_PYTHON = r"C:\Anaconda\envs\stalingrad-rl\python.exe"
DEFAULT_STATUS_FILE = os.path.join("checkpoints", "train_status.json")


class TrainingManager:
    """Singleton manager for training subprocess and RL model serving."""

    def __init__(self, checkpoints_dir="checkpoints"):
        self.checkpoints_dir = checkpoints_dir
        self.status_file = os.path.join(checkpoints_dir, "train_status.json")
        self.proc = None
        self.start_time = None
        self.active_checkpoint = None
        self.team_checkpoints = {"axis": None, "soviet": None}
        self.metadata_file = os.path.join(checkpoints_dir, "checkpoints_meta.json")
        self._checkpoint_steps_cache = {}
        self._agent_cache = {}
        os.makedirs(checkpoints_dir, exist_ok=True)

    def _get_python_exe(self):
        if os.path.isfile(CONDA_RL_PYTHON):
            return CONDA_RL_PYTHON
        return sys.executable

    def start_training(self, total_timesteps=10000, num_envs=4, lr=2.5e-4, resume_checkpoint=None, train_side="axis", opponent="heuristic", opponent_checkpoint=None):
        """Start PPO training in a background subprocess."""
        if self.is_running():
            return False, "Training is already in progress."

        python_exe = self._get_python_exe()
        cmd = [
            python_exe,
            os.path.join("rl", "train_ppo.py"),
            "--total-timesteps", str(int(total_timesteps)),
            "--num-envs", str(int(num_envs)),
            "--learning-rate", str(float(lr)),
            "--status-file", self.status_file,
            "--save-dir", self.checkpoints_dir,
            "--train-side", str(train_side),
            "--opponent", str(opponent),
        ]

        if opponent == "checkpoint" and opponent_checkpoint:
            opp_path = os.path.join(self.checkpoints_dir, opponent_checkpoint)
            if not os.path.isfile(opp_path) and os.path.isfile(opponent_checkpoint):
                opp_path = opponent_checkpoint
            if os.path.isfile(opp_path):
                cmd.extend(["--opponent-checkpoint", opp_path])

        if resume_checkpoint:
            resume_path = os.path.join(self.checkpoints_dir, resume_checkpoint)
            if not os.path.isfile(resume_path) and os.path.isfile(resume_checkpoint):
                resume_path = resume_checkpoint
            if os.path.isfile(resume_path):
                cmd.extend(["--resume-checkpoint", resume_path])

        # Initialize status file
        initial_status = {
            "status": "training",
            "step": 0,
            "total_steps": int(total_timesteps),
            "progress": 0.0,
            "sps": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "win_rate": 0.0,
            "elapsed": 0.0,
            "train_side": train_side,
            "opponent": opponent,
            "opponent_checkpoint": opponent_checkpoint,
            "resumed_from": resume_checkpoint if resume_checkpoint else None,
        }
        with open(self.status_file, "w") as f:
            json.dump(initial_status, f, indent=2)

        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=os.getcwd(),
            )
            self.start_time = time.time()
            return True, "Training started successfully."
        except Exception as exc:
            return False, f"Failed to start training: {exc}"

    def stop_training(self):
        """Gracefully terminate background training."""
        if not self.is_running():
            return False, "No training run is currently active."

        try:
            self.proc.terminate()
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass

        self.proc = None

        # Update status file
        status = self.get_status()
        status["status"] = "stopped"
        try:
            with open(self.status_file, "w") as f:
                json.dump(status, f, indent=2)
        except Exception:
            pass

        return True, "Training stopped."

    def is_running(self):
        """Check if training subprocess is alive."""
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def get_status(self):
        """Read and return current training status and progress."""
        running = self.is_running()
        status_data = {
            "status": "training" if running else "idle",
            "step": 0,
            "total_steps": 0,
            "progress": 0.0,
            "sps": 0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "win_rate": 0.0,
            "elapsed": 0.0,
            "checkpoints": self.list_checkpoints(),
            "active_checkpoint": self.active_checkpoint,
            "team_checkpoints": dict(self.team_checkpoints),
        }

        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, "r") as f:
                    file_data = json.load(f)
                    status_data.update(file_data)
            except Exception:
                pass

        if not running and status_data["status"] == "training":
            status_data["status"] = "completed"

        status_data["is_running"] = running
        status_data["checkpoints"] = self.list_checkpoints()
        status_data["checkpoint_steps"] = self.get_all_checkpoint_steps()
        status_data["active_checkpoint"] = self.active_checkpoint
        status_data["team_checkpoints"] = dict(self.team_checkpoints)
        return status_data

    def get_checkpoint_steps(self, filename):
        """Return the total steps completed by a given checkpoint filename."""
        if not filename:
            return None
        filename = os.path.basename(filename)

        full_path = os.path.join(self.checkpoints_dir, filename)
        if not os.path.isfile(full_path):
            return None

        mtime = os.path.getmtime(full_path)
        if filename in self._checkpoint_steps_cache:
            cached_mtime, cached_steps = self._checkpoint_steps_cache[filename]
            if cached_mtime == mtime:
                return cached_steps

        steps = None

        # 1. Check metadata file
        if os.path.isfile(self.metadata_file):
            try:
                with open(self.metadata_file, "r") as mf:
                    meta = json.load(mf)
                    if filename in meta and isinstance(meta[filename], dict):
                        steps = meta[filename].get("total_steps")
            except Exception:
                pass

        # 2. Check train_status.json if latest_checkpoint matches
        if steps is None and os.path.isfile(self.status_file):
            try:
                with open(self.status_file, "r") as sf:
                    sdata = json.load(sf)
                    latest_cp = sdata.get("latest_checkpoint")
                    if latest_cp and os.path.basename(latest_cp) == filename:
                        steps = sdata.get("step") or sdata.get("total_steps")
            except Exception:
                pass

        # 3. If still None, inspect .pt file
        if steps is None:
            try:
                import torch
                raw = torch.load(full_path, map_location="cpu", weights_only=True)
                if isinstance(raw, dict):
                    steps = raw.get("total_steps") or raw.get("step") or raw.get("global_step")
            except Exception:
                pass

        if steps is not None:
            self._checkpoint_steps_cache[filename] = (mtime, steps)
            self._save_checkpoint_meta(filename, steps)
            return steps

        self._checkpoint_steps_cache[filename] = (mtime, None)
        return None

    def _save_checkpoint_meta(self, filename, steps):
        try:
            meta = {}
            if os.path.isfile(self.metadata_file):
                with open(self.metadata_file, "r") as mf:
                    meta = json.load(mf)
            if filename not in meta or meta[filename].get("total_steps") != steps:
                meta[filename] = {"total_steps": steps}
                with open(self.metadata_file, "w") as mf:
                    json.dump(meta, mf, indent=2)
        except Exception:
            pass

    def get_all_checkpoint_steps(self):
        """Return dict of {filename: total_steps} for all available checkpoints."""
        steps_dict = {}
        for cp in self.list_checkpoints():
            steps = self.get_checkpoint_steps(cp)
            if steps is not None:
                steps_dict[cp] = steps
        return steps_dict

    def list_checkpoints(self):
        """Return list of saved checkpoint filenames."""
        if not os.path.isdir(self.checkpoints_dir):
            return []
        files = [
            f for f in os.listdir(self.checkpoints_dir)
            if f.endswith(".pt")
        ]
        files.sort(reverse=True)
        return files

    def set_team_checkpoint(self, team, filename):
        """Set which model checkpoint to use for a specific team."""
        if team not in ("axis", "soviet"):
            raise ValueError(f"Invalid team: {team}")
        if not filename:
            self.team_checkpoints[team] = None
            return True
        full_path = os.path.join(self.checkpoints_dir, filename)
        if not os.path.isfile(full_path):
            raise ValueError(f"Checkpoint not found: {filename}")
        self.team_checkpoints[team] = filename
        self.active_checkpoint = filename
        return True

    def get_team_checkpoint(self, team):
        """Return the active checkpoint filename for a team."""
        cp = self.team_checkpoints.get(team)
        if cp:
            return cp
        if self.active_checkpoint:
            return self.active_checkpoint
        checkpoints = self.list_checkpoints()
        if checkpoints:
            return checkpoints[0]
        return None

    def set_active_checkpoint(self, filename):
        """Set active checkpoint (backwards compatibility)."""
        self.set_team_checkpoint("axis", filename)
        self.set_team_checkpoint("soviet", filename)
        return True

    def get_agent_for_file(self, filename):
        """Load and cache RLAgent for a given checkpoint filename."""
        if not filename:
            return None
        full_path = os.path.join(self.checkpoints_dir, filename)
        if not os.path.isfile(full_path):
            return None
        if filename in self._agent_cache:
            return self._agent_cache[filename]

        try:
            from rl.agent import RLAgent
            agent = RLAgent(full_path)
            self._agent_cache[filename] = agent
            return agent
        except Exception as exc:
            print(f"Error loading RLAgent for {filename}: {exc}")
            return None

    def get_agent_for_team(self, team):
        """Return cached RLAgent for the specified team."""
        cp = self.get_team_checkpoint(team)
        if not cp:
            return None
        return self.get_agent_for_file(cp)

    def get_active_agent(self):
        """Return cached RLAgent (backwards compatibility)."""
        return self.get_agent_for_team("axis")


TRAINING_MANAGER = TrainingManager()
