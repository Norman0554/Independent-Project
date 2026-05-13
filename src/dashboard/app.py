from flask import Flask, render_template, jsonify
import os
import yaml
from datetime import datetime
import json
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
app = Flask(__name__, template_folder=TEMPLATE_DIR)

# Load configuration
with open(os.path.join(BASE_DIR, 'config', 'config.yaml')) as f:
    config = yaml.safe_load(f)


def _alerts_log_path():
    return os.path.join(BASE_DIR, config['logging']['log_path'], "alerts.log")


def _violations_path():
    return os.path.join(BASE_DIR, config['global']['output_path'], "violations.json")


def _read_alerts(limit=200):
    log_file = _alerts_log_path()
    if not os.path.exists(log_file):
        return []

    with open(log_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines[-limit:]


def _read_violations():
    path = _violations_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/alerts')
def get_alerts():
    alerts = _read_alerts(limit=20)
    return jsonify(alerts[::-1])


@app.route('/api/violations')
def get_violations():
    violations = _read_violations()
    return jsonify(violations[-100:][::-1])

@app.route('/api/stats')
def get_stats():
    alerts = _read_alerts(limit=500)
    violations = _read_violations()
    by_type = Counter(v.get("type", "UNKNOWN") for v in violations)
    last_alert = alerts[-1] if alerts else "No alerts yet"
    last_violation = violations[-1]["type"] if violations else "NONE"
    total_violations = len(violations)
    cheating_probability = min(100, total_violations * 8)

    high_risk_types = {"OBJECT_DETECTED", "MULTIPLE_FACES", "VOICE_DETECTED"}
    current_activity = "Normal"
    if last_violation in high_risk_types:
        current_activity = "High Risk"
    elif total_violations > 0:
        current_activity = "Suspicious"

    return jsonify({
        'current_time': datetime.now().strftime("%H:%M:%S"),
        'total_alerts': len(alerts),
        'total_violations': total_violations,
        'current_activity': current_activity,
        'cheating_probability': cheating_probability,
        'last_alert': last_alert,
        'by_type': dict(by_type),
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
