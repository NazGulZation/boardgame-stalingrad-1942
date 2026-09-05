"""Flask web app for the Stalingrad 1942 board game (2 players, hot-seat).

Thin controller: validates nothing itself, just forwards to the engine and
converts ValueError into HTTP 400 with an error message.
"""

from flask import Flask, jsonify, render_template, request

from game import Game

app = Flask(__name__)
GAME = Game()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def state():
    return jsonify(GAME.to_dict())


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
    return jsonify(GAME.to_dict())


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


@app.route("/api/reset", methods=["POST"])
def reset():
    global GAME
    GAME = Game()
    return jsonify(GAME.to_dict())


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
