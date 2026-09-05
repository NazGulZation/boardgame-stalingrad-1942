"""Flask web app for the Stalingrad 1942 board game (2 players, hot-seat).

Thin controller: validates nothing itself, just forwards to the engine and
converts ValueError into HTTP 400 with an error message. AI sides are chosen
with /api/set_ai and driven with /api/ai_turn.
"""

from flask import Flask, jsonify, render_template, request

from ai import play_turn
from game import Game
from units import TEAM_NAMES

app = Flask(__name__)
GAME = Game()
AI_SIDES = {"axis": False, "soviet": False}


def _state():
    data = GAME.to_dict()
    data["ai"] = dict(AI_SIDES)
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


@app.route("/api/ai_turn", methods=["POST"])
def ai_turn():
    if not AI_SIDES.get(GAME.turn):
        return jsonify(
            {"error": "The %s is not computer-controlled."
                      % TEAM_NAMES[GAME.turn]}), 400
    try:
        play_turn(GAME, GAME.turn)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_state())


@app.route("/api/reset", methods=["POST"])
def reset():
    global GAME
    GAME = Game()
    return jsonify(_state())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
