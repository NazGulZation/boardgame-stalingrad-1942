"""Flask web app for the Stalingrad 1942 board game (2 players, hot-seat).

Thin controller: validates nothing itself, just forwards to the engine and
converts ValueError into HTTP 400 with an error message. AI sides are chosen
with /api/set_ai and driven with /api/ai_turn (supporting both heuristic and RL).
Training management endpoints are available under /api/training/*.
"""

from flask import Flask, jsonify, render_template, request

from ai import play_turn
from game import Game
from units import TEAM_NAMES
from rl.train_manager import TRAINING_MANAGER

app = Flask(__name__)
GAME = Game()
AI_SIDES = {"axis": False, "soviet": False}
AI_TYPES = {"axis": "heuristic", "soviet": "heuristic"}


def _state():
    data = GAME.to_dict()
    data["ai"] = dict(AI_SIDES)
    data["ai_types"] = dict(AI_TYPES)
    data["training"] = TRAINING_MANAGER.get_status()
    return data


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(_state())


@app.route("/api/legal_moves", methods=["POST"])
def legal_moves():
    data = request.get_json(force=True, silent=True) or {}
    try:
        return jsonify({
            "moves": GAME.legal_moves(data.get("unit_id", "")),
            "targets": GAME.attackable(data.get("unit_id", "")),
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


def _run_action(action, *args):
    try:
        action(*args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_state())


@app.route("/api/move", methods=["POST"])
def move():
    data = request.get_json(force=True, silent=True) or {}
    return _run_action(GAME.move, data.get("unit_id", ""),
                       data.get("x"), data.get("y"))


@app.route("/api/attack", methods=["POST"])
def attack():
    data = request.get_json(force=True, silent=True) or {}
    return _run_action(GAME.attack, data.get("attacker_id", ""),
                       data.get("target_id", ""))


@app.route("/api/end_turn", methods=["POST"])
def end_turn():
    return _run_action(GAME.end_turn)


@app.route("/api/set_ai", methods=["POST"])
def set_ai():
    global AI_SIDES
    data = request.get_json(force=True, silent=True) or {}
    for team in ("axis", "soviet"):
        if team in data:
            AI_SIDES[team] = bool(data[team])
    return jsonify(_state())


@app.route("/api/set_ai_type", methods=["POST"])
def set_ai_type():
    global AI_TYPES
    data = request.get_json(force=True, silent=True) or {}
    for team in ("axis", "soviet"):
        if team in data and data[team] in ("heuristic", "rl"):
            AI_TYPES[team] = data[team]
    if "checkpoint" in data and data["checkpoint"]:
        try:
            TRAINING_MANAGER.set_active_checkpoint(data["checkpoint"])
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    return jsonify(_state())


@app.route("/api/ai_turn", methods=["POST"])
def ai_turn():
    if not AI_SIDES.get(GAME.turn):
        return jsonify(
            {"error": "The %s is not computer-controlled."
                      % TEAM_NAMES[GAME.turn]}), 400
    try:
        if AI_TYPES.get(GAME.turn) == "rl":
            agent = TRAINING_MANAGER.get_active_agent()
            if agent is not None:
                agent.play_turn(GAME, GAME.turn)
            else:
                play_turn(GAME, GAME.turn)
        else:
            play_turn(GAME, GAME.turn)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_state())


@app.route("/api/training/status", methods=["GET"])
def training_status():
    return jsonify(TRAINING_MANAGER.get_status())


@app.route("/api/training/start", methods=["POST"])
def training_start():
    data = request.get_json(force=True, silent=True) or {}
    total_timesteps = data.get("total_timesteps", 10000)
    num_envs = data.get("num_envs", 4)
    lr = data.get("learning_rate", 2.5e-4)

    success, message = TRAINING_MANAGER.start_training(
        total_timesteps=total_timesteps, num_envs=num_envs, lr=lr
    )
    if not success:
        return jsonify({"error": message}), 400
    return jsonify({"message": message, "training": TRAINING_MANAGER.get_status()})


@app.route("/api/training/stop", methods=["POST"])
def training_stop():
    success, message = TRAINING_MANAGER.stop_training()
    if not success:
        return jsonify({"error": message}), 400
    return jsonify({"message": message, "training": TRAINING_MANAGER.get_status()})


@app.route("/api/training/select_model", methods=["POST"])
def training_select_model():
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("model", "")
    try:
        TRAINING_MANAGER.set_active_checkpoint(filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"training": TRAINING_MANAGER.get_status()})


@app.route("/api/reset", methods=["POST"])
def reset():
    global GAME
    GAME = Game()
    return jsonify(_state())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
