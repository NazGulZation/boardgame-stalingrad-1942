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
        self._cached_agent = None
        self._cached_path = None
        os.makedirs(checkpoints_dir, exist_ok=True)

    def _get_python_exe(self):
        if os.path.isfile(CONDA_RL_PYTHON):
            return CONDA_RL_PYTHON
        return sys.executable

    def start_training(self, total_timesteps=10000, num_envs=4, lr=2.5e-4):
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
        ]

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
        return status_data

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

    def set_active_checkpoint(self, filename):
        """Set which model checkpoint to use for RL gameplay."""
        full_path = os.path.join(self.checkpoints_dir, filename)
        if not os.path.isfile(full_path):
            raise ValueError(f"Checkpoint not found: {filename}")
        self.active_checkpoint = filename
        self._cached_agent = None
        self._cached_path = None
        return True

    def get_active_agent(self):
        """Return cached RLAgent for the active checkpoint."""
        if not self.active_checkpoint:
            checkpoints = self.list_checkpoints()
            if checkpoints:
                self.active_checkpoint = checkpoints[0]
            else:
                return None

        full_path = os.path.join(self.checkpoints_dir, self.active_checkpoint)
        if self._cached_agent is not None and self._cached_path == full_path:
            return self._cached_agent

        try:
            from rl.agent import RLAgent
            self._cached_agent = RLAgent(full_path)
            self._cached_path = full_path
            return self._cached_agent
        except Exception as exc:
            print(f"Error loading RLAgent: {exc}")
            return None


TRAINING_MANAGER = TrainingManager()
