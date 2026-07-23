"""
server.py — Flask web server.

Routes:
  GET  /  or  /view        → dashboard HTML
  GET  /stream             → MJPEG camera feed
  GET  /status             → robot telemetry JSON
  GET  /log_json           → last 60 log entries
  GET  /snapshot           → save + return info JSON
  POST /stop               → emergency stop
  POST /command            → send natural language command to ROSA
  POST /vlm/switch         → hot-swap VLM backend
  GET  /vlm/status         → backend info JSON
"""

import time
import logging

from flask import Flask, Response, request, jsonify, render_template_string

from state import state, interrupt_flag
from logger import log, log_entries
from vlm.manager import vlm_mgr
from motion import _stop
from stream import generate_stream
from agent import agent, rosa_lock

# ── Flask app ────────────────────────────────────────────────────
app = Flask(__name__)

# Silence noisy werkzeug GET-per-frame logs
_werkzeug_log = logging.getLogger("werkzeug")
_werkzeug_log.setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════
@app.route("/stream")
def stream():
    return Response(generate_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/stop", methods=["POST", "GET"])
def emergency_stop():
    interrupt_flag.set()
    _stop()
    state["status"] = "idle"
    state["target"] = ""
    # Release lock immediately so next command doesn't wait
    try:
        rosa_lock.release()
    except RuntimeError:
        pass   # wasn't held, fine
    log("⚡ EMERGENCY STOP — lock released", "WARN")
    return jsonify({"response": "⚡ Robot stopped immediately."})
# def emergency_stop():
#     interrupt_flag.set()
#     _stop()
#     state["status"] = "idle"
#     state["target"] = ""
#     log("⚡ EMERGENCY STOP", "WARN")
#     return jsonify({"response": "⚡ Robot stopped immediately."})


@app.route("/vlm/switch", methods=["POST"])
def vlm_switch():
    """Switch VLM backend at runtime. Body: {"mode": "local"} or {"mode": "api"}"""
    data = request.get_json(silent=True) or {}
    mode = data.get("mode") or request.form.get("mode", "")
    if not mode:
        return jsonify({"error": "Provide {mode: 'api'|'local'|'remote'}"}), 400
    result = vlm_mgr.switch(mode)
    log(f"VLM switch: {result}", "VLM")
    return jsonify({"result": result, "active": vlm_mgr.active_name})



@app.route("/vlm/model/list")
def vlm_model_list():
    from vlm.backends import API_MODELS
    return jsonify({
        "models":       API_MODELS,
        "active_model": vlm_mgr.active_model,
        "active_backend": vlm_mgr.active_name,
    })

@app.route("/vlm/model/switch", methods=["POST"])
def vlm_model_switch():
    data  = request.get_json(silent=True) or {}
    model = data.get("model", "").strip()
    if not model:
        return jsonify({"error": "Provide {model: 'claude-sonnet-4-6'}"}), 400
    result = vlm_mgr.switch_model(model)
    log(f"VLM model switch: {result}", "VLM")
    return jsonify({"result": result, "active_model": vlm_mgr.active_model})


@app.route("/vlm/status")
def vlm_status():
    return jsonify(vlm_mgr.status())


@app.route("/command", methods=["POST"])
def command():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "no text"}), 400

    # Hard stop words bypass ROSA entirely
    STOP_WORDS = {"stop", "halt", "cancel", "abort", "freeze", "wait", "pause", "enough"}
    if any(w in text.lower() for w in STOP_WORDS):
        interrupt_flag.set()
        _stop()
        state["status"] = "idle"
        state["target"] = ""
        log("⚡ INTERRUPT received", "WARN")
        return jsonify({"response": "Stopped. Robot halted immediately."})

    # If agent is busy, interrupt and wait
    _lock_acquired = rosa_lock.acquire(blocking=False)

    if not _lock_acquired:
        log(f"Interrupting current task for: {text}", "WARN")
        interrupt_flag.set()
        _stop()
        time.sleep(0.5)

        _lock_acquired = rosa_lock.acquire(blocking=True, timeout=5)
        if not _lock_acquired:
            state["status"] = "idle"
            return jsonify({"error": "Agent busy — previous command did not finish cleanly"}), 503
    interrupt_flag.clear()
    log(f'Command: "{text}" [{vlm_mgr.active_name}]', "VROSA")
    try:
        state["status"] = "processing"
        response = agent.invoke(text)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        state["status"] = "idle"
        time.sleep(0.05)   # tiny settle so chain fully exits before lock release
        if _lock_acquired:
            try:
                rosa_lock.release()
            except RuntimeError:
                pass
    # finally:
    #     state["status"] = "idle"
    #     if _lock_acquired:
    #         rosa_lock.release()
    #     # rosa_lock.release()


@app.route("/status")
def status():
    lidar = state["lidar"]
    return jsonify({
        "x":             round(state["x"], 3),
        "y":             round(state["y"], 3),
        "heading":       round(state["yaw"], 1),
        "lidar_closest": round(lidar[0]["dist"], 2) if lidar else None,
        "camera":        state["image_raw"] is not None,
        "status":        state["status"],
        "last_vlm":      state["last_vlm"][:100],
        "target":        state["target"],
        "vlm_backend":   vlm_mgr.active_name,
    })


@app.route("/log_json")
def log_json():
    return jsonify({"entries": log_entries[-60:]})


@app.route("/snapshot")
def snapshot():
    img = state["image_raw"]
    if not img:
        return "No image", 503
    path = "/home/caim/snapshot.png"
    img.save(path)
    return jsonify({"saved": path, "size": f"{img.size[0]}x{img.size[1]}"})



@app.route("/detections")
def detections_json():
    dets = state.get("detections", [])
    perc_ok = state.get("perc_ready", False)
    try:
        from perception import _yolo_dets, _gdino_dets, _yolo_key as yk
        n_yolo  = len(_yolo_dets)
        n_gdino = len(_gdino_dets)
        yolo_active = yk
    except Exception:
        n_yolo = n_gdino = 0
        yolo_active = state.get("yolo_model", "yolo26n")
    return jsonify({
        "count":       len(dets),
        "n_yolo":      n_yolo,
        "n_gdino":     n_gdino,
        "yolo_model":  yolo_active,
        "has_lidar":   state.get("has_lidar", False),
        "show_dets":   state.get("show_dets", False),
        "perc_ready":  perc_ok,
        "detections": [
            {
                "label":  d["label"],
                "dist":   round(d["dist"], 2),
                "sector": d["sector"],
                "conf":   d["conf"],
                "source": d.get("source", "yolo"),
                "bbox":   d["bbox"],
            }
            for d in sorted(dets, key=lambda x: x["dist"])
        ],
    })



@app.route("/detections/toggle", methods=["POST"])
def detections_toggle():
    state["show_dets"] = not state.get("show_dets", False)
    log(f"Detection overlay: {'ON' if state['show_dets'] else 'OFF'}", "PERC")
    return jsonify({"show_dets": state["show_dets"]})



@app.route("/yolo/models")
def yolo_models():
    from perception import YOLO_MODELS, _yolo_key
    return jsonify({
        "current": _yolo_key,
        "models":  {k: v for k, v in YOLO_MODELS.items()},
    })

@app.route("/yolo/switch", methods=["POST"])
def yolo_switch():
    data = request.get_json(silent=True) or {}
    key  = data.get("model", "").strip()
    if not key:
        return jsonify({"error": "Provide {model: 'yolo11n'}"}), 400
    from perception import switch_yolo
    result = switch_yolo(key)
    log(f"YOLO switch requested: {key}", "PERC")
    return jsonify({"result": result, "model": key})



@app.route("/perc/config", methods=["GET"])
def perc_config_get():
    return jsonify({
        "use_lidar":  state.get("use_lidar",  True),
        "use_depth":  state.get("use_depth",  True),
        "use_yolo":   state.get("use_yolo",   True),
        "use_gdino":  state.get("use_gdino",  True),
        "has_lidar":  state.get("has_lidar",  False),
        "perc_ready": state.get("perc_ready", False),
    })

@app.route("/perc/config", methods=["POST"])
def perc_config_set():
    data = request.get_json(silent=True) or {}
    changed = []
    for key in ("use_lidar", "use_depth", "use_yolo", "use_gdino"):
        if key in data:
            state[key] = bool(data[key])
            changed.append(f"{key}={'ON' if state[key] else 'OFF'}")
    if changed:
        log(f"Perc config: {', '.join(changed)}", "PERC")
    return jsonify({
        "use_lidar":  state["use_lidar"],
        "use_depth":  state["use_depth"],
        "use_yolo":   state["use_yolo"],
        "use_gdino":  state["use_gdino"],
    })


import os as _os

@app.route("/logo.png")
def serve_logo():
    """Serve the lab logo from project directory."""
    from flask import send_file, abort
    logo_path = _os.path.join(_os.path.dirname(__file__), "logo.png")
    if _os.path.exists(logo_path):
        return send_file(logo_path, mimetype="image/png")
    abort(404)

# Server-side path trail (world frame absolute coords)
_path_trail = []   # [{x,y}]
_PATH_MAX   = 300
_path_last  = [0.0, 0.0]

@app.route("/rviz_data")
def rviz_data():
    """Lightweight JSON for the browser-side RViz canvas."""
    global _path_trail, _path_last

    rx = round(state.get("x", 0.0), 3)
    ry = round(state.get("y", 0.0), 3)

    # Append to trail when moved > 0.05m
    if (_path_trail == [] or
        (rx - _path_last[0])**2 + (ry - _path_last[1])**2 > 0.0025):
        _path_trail.append({"x": rx, "y": ry})
        _path_last = [rx, ry]
        if len(_path_trail) > _PATH_MAX:
            _path_trail = _path_trail[-_PATH_MAX:]

    lidar = state.get("lidar", [])
    dets  = state.get("detections", [])
    step  = max(1, len(lidar) // 800)
    pts   = [{"x": round(p["x"], 2), "y": round(p["y"], 2),
               "d": round(p["dist"], 1)}
              for p in lidar[::step]]

    return jsonify({
        "x":        rx,
        "y":        ry,
        "yaw":      round(state.get("yaw", 0.0), 1),
        "status":   state.get("status", "idle"),
        "lidar":    pts,
        "dets":     [{"label": d["label"], "dist": d["dist"],
                      "sector": d["sector"], "source": d.get("source","yolo")}
                     for d in dets[:20]],
        "has_lidar": state.get("has_lidar", False),
        "trail":    _path_trail[-150:],   # last 150 pts to client
    })

@app.route("/rviz_data/reset_trail", methods=["POST"])
def rviz_reset_trail():
    global _path_trail, _path_last
    _path_trail = []
    _path_last  = [0.0, 0.0]
    return jsonify({"cleared": True})


# ════════════════════════════════════════════════════════════
# DASHBOARD HTML
# ════════════════════════════════════════════════════════════
DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
  <title>V-ROSA Live</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; }
    body { background:#F4F0EA; color:#1C1C2E; font-family:monospace; }

    .top { display:flex; align-items:center; background:#FFFFFF;
           padding:12px 20px; border-bottom:1px solid #E2D9CC; box-shadow:0 1px 4px rgba(0,0,0,0.06); gap:8px; }
    .top h1 { color:#0369A1; font-size:20px; margin-right:12px; }
    .badge { background:#EFF6FF; color:#0369A1; border:1px solid #BAD9F5;
             padding:3px 10px; border-radius:4px; font-size:12px; }
    .badge.local  { color:#166534; border-color:#86EFAC; background:#F0FDF4; }
    .badge.loading{ color:#92400E; border-color:#FCD34D; background:#FFFBEB; }

    .main { display:flex; height:calc(100vh - 54px); }
    .stream-panel { flex:1; display:flex; flex-direction:column;
                    padding:12px; gap:10px; }
    .stream-panel img { width:100%; border-radius:8px; border:1px solid #D6CEC4; box-shadow:0 2px 8px rgba(0,0,0,0.08); }
    .cmd-panel { width:400px; display:flex; flex-direction:column;
                 padding:12px; gap:10px; border-left:1px solid #E2D9CC; overflow-y:auto; }

    .card { background:#FFFFFF; border:1px solid #E2D9CC;
            border-radius:6px; padding:14px; }
    .card h3 { color:#0369A1; font-size:13px; margin-bottom:10px; }

    input[type=text] {
      width:100%; background:#FAFAF8; border:1px solid #D6CEC4;
      color:#1C1C2E; padding:10px; border-radius:4px;
      font-family:monospace; font-size:14px;
    }
    button { background:#0369A1; color:#fff; border:none; padding:10px 16px;
             border-radius:4px; cursor:pointer; font-weight:bold;
             font-size:13px; width:100%; margin-top:8px; }
    button:hover { background:#0284C7; }
    button.secondary { background:#F0F9FF; color:#0369A1;
                       border:1px solid #BAD9F5; }
    button.danger { background:#FF6B6B; color:#fff; }
    button.danger:hover { background:#FF3B3B; }
    button.active-btn { background:#16A34A; color:#fff; }
    button.inactive-btn { background:#F0FDF4; color:#16A34A;
                          border:1px solid #86EFAC; }

    .stat { display:flex; justify-content:space-between; padding:5px 0;
            border-bottom:1px solid #EDE8E0; font-size:13px; }
    .stat:last-child { border:none; }
    .stat-label { color:#78716C; }
    .stat-val   { color:#0369A1; }

    #response { background:#F8F7F4; border:1px solid #E2D9CC; border-radius:4px;
                padding:10px; min-height:80px; font-size:12px; color:#166534;
                white-space:pre-wrap; word-wrap:break-word; margin-top:8px; }

    .log-box { background:#FAFAF8; border:1px solid #E2D9CC; border-radius:4px;
               padding:8px; height:160px; overflow-y:auto; font-size:11px; }
    .log-ROBOT  { color:#92400E; }
    .log-VLM    { color:#0369A1; }
    .log-VROSA  { color:#7C3AED; }
    .log-INFO   { color:#78716C; }
    .log-WARN   { color:#B45309; }
    .log-ROS    { color:#0F766E; }

    .quick-btn { background:#F4F0EA; color:#1C1C2E; border:1px solid #D6CEC4;
                 padding:6px 10px; border-radius:4px; cursor:pointer;
                 font-size:12px; margin:2px; font-family:monospace; }
    .quick-btn:hover { border-color:#0369A1; color:#0369A1; background:#EFF6FF; }

    #status-dot { width:10px; height:10px; border-radius:50%; background:#0369A1;
                  display:inline-block; margin-right:6px;
                  animation:pulse 2s infinite; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

    .vlm-toggle { display:flex; gap:6px; margin-top:8px; }
    .vlm-toggle button { flex:1; margin:0; font-size:12px; padding:8px; }

    .vlm-info { background:#F8F7F4; border:1px solid #E2D9CC; border-radius:4px;
                padding:8px; margin-top:8px; font-size:11px; color:#546E7A; }
    .vlm-info .key   { color:#78716C; }
    .vlm-info .value { color:#166534; float:right; }

    .det-row { display:flex; align-items:center; padding:4px 0;
               border-bottom:1px solid #EDE8E0; font-size:11px; gap:6px; }
    .det-row:last-child { border:none; }
    .det-label { flex:1; color:#1C1C2E; font-weight:bold; }
    .det-dist  { color:#0369A1; min-width:40px; text-align:right; }
    .det-sector{ color:#78716C; min-width:44px; text-align:center; }
    .det-conf  { color:#78716C; min-width:36px; text-align:right; }
    .det-src-yolo      { width:6px; height:6px; border-radius:50%;
                         background:#00D4FF; display:inline-block; }
    .det-src-grounding { width:6px; height:6px; border-radius:50%;
                         background:#A5D6A7; display:inline-block; }
    .det-box { background:#F8F7F4; border:1px solid #E2D9CC; border-radius:4px;
               padding:6px 8px; max-height:180px; overflow-y:auto; margin-top:6px; }
    #det-count-badge { background:#EFF6FF; color:#0369A1; border:1px solid #BAD9F5;
                       padding:2px 8px; border-radius:4px; font-size:11px; }

    .det-toggle-btn {
      display:flex; align-items:center; justify-content:center; gap:8px;
      width:100%; padding:9px 12px; border-radius:4px; cursor:pointer;
      font-family:monospace; font-size:13px; font-weight:bold;
      border:2px solid #D6CEC4; background:#FFFFFF; color:#78716C;
      transition: all 0.15s;
    }
    .det-toggle-btn.on {
      border-color:#16A34A; background:#F0FDF4; color:#16A34A;
    }
    .det-toggle-btn .dot {
      width:9px; height:9px; border-radius:50%; background:#D6CEC4;
      transition: background 0.15s;
    }
    .det-toggle-btn.on .dot { background:#A5D6A7; box-shadow:0 0 6px #A5D6A7; }

    .det-table { width:100%; border-collapse:collapse; font-size:11px; }
    .det-table th { color:#78716C; font-weight:normal; padding:3px 4px;
                    border-bottom:1px solid #EDE8E0; text-align:left; }
    .det-table th.r { text-align:right; }
    .det-table td { padding:4px 4px; border-bottom:1px solid #F0EBE2;
                    vertical-align:middle; }
    .det-table tr:last-child td { border:none; }
    .det-table tr:hover td { background:#F0F9FF; }
    .det-dist-val { color:#00D4FF; font-weight:bold; text-align:right; }
    .det-sector-val { color:#78716C; text-align:center; }
    .det-conf-val { color:#78716C; text-align:right; }
    .det-label-val { color:#1C1C2E; }
    .src-dot-yolo      { width:7px; height:7px; border-radius:50%;
                         background:#00D4FF; display:inline-block; }
    .src-dot-grounding { width:7px; height:7px; border-radius:50%;
                         background:#A5D6A7; display:inline-block; }
    .det-empty { color:#78716C; font-size:11px; padding:8px 4px; }
    .det-scroll { max-height:200px; overflow-y:auto; }
    .det-loading { color:#78716C; font-size:11px; font-style:italic; padding:6px 0; }

    .yolo-select-wrap { margin-top:8px; position:relative; }
    .yolo-select {
      width:100%; background:#FAFAF8; border:1px solid #D6CEC4;
      color:#0369A1; padding:8px 10px; border-radius:4px;
      font-family:monospace; font-size:12px; cursor:pointer;
      appearance:none; -webkit-appearance:none;
    }
    .yolo-select:focus { outline:none; border-color:#0369A1; }
    .yolo-select-wrap::after {
      content:'▾'; position:absolute; right:10px; top:50%;
      transform:translateY(-50%); color:#78716C; pointer-events:none;
    }
    .yolo-status { font-size:10px; color:#78716C; margin-top:5px;
                   min-height:14px; }
    .yolo-status.loading { color:#FFE082; }
    .yolo-status.ready   { color:#A5D6A7; }

    .vlm-model-wrap { margin-top:8px; position:relative; }
    .vlm-model-select {
      width:100%; background:#FAFAF8; border:1px solid #D6CEC4;
      color:#0369A1; padding:8px 10px; border-radius:4px;
      font-family:monospace; font-size:12px; cursor:pointer;
      appearance:none; -webkit-appearance:none;
    }
    .vlm-model-select:focus { outline:none; border-color:#0369A1; }
    .vlm-model-wrap::after {
      content:'▾'; position:absolute; right:10px; top:50%;
      transform:translateY(-50%); color:#78716C; pointer-events:none;
    }
    .vlm-model-status { font-size:10px; margin-top:5px; min-height:14px; color:#78716C; }
    .vlm-model-status.ok  { color:#A5D6A7; }
    .vlm-model-status.err { color:#FF6B6B; }

    .perc-grid { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
    .perc-toggle {
      display:flex; align-items:center; justify-content:space-between;
      padding:7px 10px; border-radius:4px; cursor:pointer;
      border:1px solid #D6CEC4; background:#FAFAF8;
      font-family:monospace; font-size:11px; color:#78716C;
      transition:all 0.15s; user-select:none;
    }
    .perc-toggle.on  { border-color:#16A34A; color:#16A34A; background:#F0FDF4; }
    .perc-toggle.off { border-color:#FECACA; color:#B91C1C; background:#FFF5F5; }
    .perc-toggle .pt-dot {
      width:7px; height:7px; border-radius:50%; background:#D6CEC4;
      transition:background 0.15s;
    }
    .perc-toggle.on .pt-dot { background:#16A34A; box-shadow:0 0 5px #A5D6A7; }
    .perc-toggle:hover { border-color:#0369A1; color:#0369A1; }
    .perc-hint { font-size:10px; color:#78716C; margin-top:6px; line-height:1.4; }
    .perc-hint span { color:#16A34A; }

    #rviz-canvas {
      width:100%; aspect-ratio:1/1; border-radius:8px;
      border:1px solid #D6CEC4; background:#F8F7F4;
      display:block;
    }
    .rviz-wrap { position:relative; }
    .rviz-legend {
      display:flex; gap:10px; flex-wrap:wrap;
      font-size:10px; color:#78716C; margin-top:5px;
    }
    .rviz-legend span { display:flex; align-items:center; gap:4px; }
    .rviz-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  </style>
</head>
<body>
<div class="top">
  <span id="status-dot"></span>
  <!-- Lab logo -->
  <img src="/logo.png" alt="CAiM"
       onerror="this.style.display='none'"
       style="height:32px; margin-right:4px; object-fit:contain;">
  <!-- Lab name -->
  <div style="display:flex; flex-direction:column; margin-right:16px; line-height:1.1;">
    <span style="color:#0369A1; font-size:15px; font-weight:bold; letter-spacing:1px;">CAiM</span>
    <span style="color:#78716C; font-size:9px;">Center for Autonomous Intelligence &amp; e-Mobility</span>
  </div>
  <div style="width:1px; height:28px; background:#E2D9CC; margin-right:12px;"></div>
  <h1 style="font-size:18px;">V-ROSA</h1>
  <span class="badge">Nova Carter</span>
  <span class="badge">Isaac Sim</span>
  <span class="badge" id="top-status">idle</span>
  <span class="badge" id="top-vlm-badge">VLM: API</span>
  <button id="det-top-btn" class="det-toggle-btn" onclick="toggleDetections()"
          style="margin-left:auto; width:auto; padding:5px 14px; font-size:12px;">
    <span class="dot"></span>
    <span id="det-top-label">Detections OFF</span>
    <span id="det-top-count" style="color:#546E7A;font-size:11px;"></span>
  </button>
</div>

<div class="main">
  <div class="stream-panel">
    <img id="feed" src="/stream" alt="Live Camera">

    <!-- Map + Quick Commands side by side -->
    <div style="display:flex; gap:10px; min-height:0;">

      <!-- Mini RViz -->
      <div class="card" style="flex:0 0 340px; min-width:0; padding:10px;">
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:6px;">
          <h3 style="margin:0; font-size:12px;">🗺 Live Map</h3>
          <div style="display:flex; align-items:center; gap:5px;">
            <span style="font-size:9px; color:#78716C;" id="rviz-pts-count">—</span>
            <button onclick="clearRvizTrail()" title="Clear trail"
                    style="width:auto; margin:0; padding:2px 7px; font-size:9px;
                           background:#F4F0EA; color:#78716C; border:1px solid #D6CEC4;">
              🗑
            </button>
          </div>
        </div>
        <div class="rviz-wrap">
          <canvas id="rviz-canvas" width="320" height="320"></canvas>
        </div>
        <div class="rviz-legend" style="margin-top:4px; gap:6px;">
          <span><span class="rviz-dot" style="background:#DC2626;"></span>&lt;2m</span>
          <span><span class="rviz-dot" style="background:#F97316;"></span>2-4m</span>
          <span><span class="rviz-dot" style="background:#0369A1;"></span>4-8m</span>
          <span><span class="rviz-dot" style="background:#16A34A;"></span>DINO</span>
          <span><span class="rviz-dot" style="background:#0369A1;"></span>YOLO</span>
        </div>
      </div>

      <!-- Quick Commands -->
      <div class="card" style="flex:1; min-width:0;">
        <h3>Quick Commands</h3>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px;">
          <button class="quick-btn" onclick="cmd('what do you see?')">👁 Describe</button>
          <button class="quick-btn" onclick="cmd('check path ahead')">🔍 Path</button>
          <button class="quick-btn" onclick="cmd('look around')">🔄 Look Around</button>
          <button class="quick-btn" onclick="cmd('save snapshot')">📸 Snapshot</button>
          <button class="quick-btn" onclick="cmd('where are you?')">📍 Position</button>
          <button class="quick-btn" onclick="cmd('stop')">⏹ Stop</button>
          <button class="quick-btn" onclick="cmd('move forward 1 meter')">↑ Fwd 1m</button>
          <button class="quick-btn" onclick="cmd('move forward -1 meter')">↓ Back 1m</button>
          <button class="quick-btn" onclick="cmd('rotate 45 degrees')">↺ L 45°</button>
          <button class="quick-btn" onclick="cmd('rotate -45 degrees')">↻ R 45°</button>
        </div>
      </div>
    </div>
  </div>

  <div class="cmd-panel">

    <!-- VLM Backend Selector -->
    <div class="card">
      <h3>⚙ VLM Backend</h3>
      <div class="vlm-toggle">
        <button id="btn-api"    class="active-btn"   onclick="switchVLM('api')">
          ☁ Cloud API
        </button>
        <button id="btn-local"  class="inactive-btn" onclick="switchVLM('local')"
                style="display:none">
          🖥 Local
        </button>
        <button id="btn-remote" class="inactive-btn" onclick="switchVLM('remote')"
                style="display:none; color:#FFE082; border-color:#FFE082;">
          ⚡ A100
        </button>
      </div>
      <div class="vlm-info" id="vlm-info">Loading...</div>
      <div style="border-top:1px solid #1a2332; margin-top:10px; padding-top:10px;">
        <div style="font-size:11px; color:#546E7A; margin-bottom:4px;">🧠 Model (API backend)</div>
        <div class="vlm-model-wrap">
            <select class="vlm-model-select" id="vlm-model-select"
                    onchange="switchVLMModel(this.value)">
            <optgroup label="── OpenAI (Free: GPT-5 Nano) ──">
                <option value="gpt-5-nano">⭐ GPT-5 Nano — FREE, fast</option>
                <option value="gpt-5-mini">GPT-5 Mini</option>
                <option value="gpt-5">GPT-5 — Flagship</option>
                <option value="gpt-4.1">GPT-4.1</option>
                <option value="gpt-4o">GPT-4o</option>
                <option value="gpt-4o-mini">GPT-4o Mini — Cheap</option>
            </optgroup>
            <optgroup label="── Claude (Anthropic) ──">
                <option value="claude-sonnet-4-6">Claude Sonnet 4.6 — Default</option>
                <option value="claude-opus-4-6">Claude Opus 4.6 — Best</option>
                <option value="claude-haiku-4-5-20251001">Claude Haiku 4.5 — Fastest</option>
                <option value="claude-4.7-opus">Claude 4.7 Opus</option>
            </optgroup>
            <optgroup label="── Gemini (Google) ──">
                <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                <option value="gemini-3-flash">Gemini 3 Flash</option>
                <option value="gemini-3.1-pro">Gemini 3.1 Pro</option>
            </optgroup>
            <optgroup label="── Meta / xAI ──">
                <option value="llama-4-maverick">Llama 4 Maverick</option>
                <option value="grok-4">Grok 4</option>
                <option value="grok-4.1-fast">Grok 4.1 Fast</option>
            </optgroup>
            <optgroup label="── Korean Models (MindLogic Gateway) ──">
                <option value="sait-3-pro">SAIT 3 Pro — MindLogic</option>
                <option value="k-exaone">K-EXAONE — LG AI</option>
                <option value="solar-pro-3">Solar Pro 3 — Upstage</option>
            </optgroup>
            <optgroup label="── Other ──">
                <option value="sonar-pro">Sonar Pro — Perplexity</option>
            </optgroup>
            </select>
        </div>
        <div class="vlm-model-status" id="vlm-model-status">Loading...</div>
      </div>
    </div>

    <!-- Telemetry -->
    <div class="card">
      <h3>Robot Telemetry</h3>
      <div class="stat"><span class="stat-label">Position X</span>
        <span class="stat-val" id="s-x">—</span></div>
      <div class="stat"><span class="stat-label">Position Y</span>
        <span class="stat-val" id="s-y">—</span></div>
      <div class="stat"><span class="stat-label">Heading</span>
        <span class="stat-val" id="s-hdg">—</span></div>
      <div class="stat"><span class="stat-label">LiDAR Closest</span>
        <span class="stat-val" id="s-lidar">—</span></div>
      <div class="stat"><span class="stat-label">Camera</span>
        <span class="stat-val" id="s-cam">—</span></div>
      <div class="stat"><span class="stat-label">Status</span>
        <span class="stat-val" id="s-status">—</span></div>
    </div>

    <!-- Command input -->
    <div class="card">
      <h3>Command V-ROSA</h3>
      <input type="text" id="cmd-input"
             placeholder="e.g. navigate to the shelf..."
             onkeydown="if(event.key==='Enter') sendCmd()">
      <div style="display:flex; gap:8px; margin-top:8px;">
        <button onclick="sendCmd()" style="flex:1; margin:0;">▶ Send</button>
        <button onclick="emergencyStop()"
                style="flex:0 0 80px; margin:0; background:#FF6B6B; color:#fff;">
          ⏹ STOP
        </button>
      </div>
      <div id="response">Response will appear here...</div>
    </div>

    <!-- Navigate to object -->
    <div class="card">
      <h3>Navigate to Object</h3>
      <input type="text" id="nav-input" placeholder="e.g. shelf, red box, door..."
             onkeydown="if(event.key==='Enter') navTo()">
      <button onclick="navTo()">🎯 Navigate To Object</button>
      <button class="secondary" onclick="findObj()">🔍 Find Object First</button>
    </div>




    <!-- Perception Sources -->
    <div class="card">
      <h3>⚙ Perception Sources</h3>
      <div class="perc-grid">
        <div class="perc-toggle on" id="pt-lidar" onclick="togglePerc('use_lidar')">
          <span>📡 LiDAR</span>
          <span class="pt-dot"></span>
        </div>
        <div class="perc-toggle on" id="pt-depth" onclick="togglePerc('use_depth')">
          <span>🌊 Depth (DA-V2)</span>
          <span class="pt-dot"></span>
        </div>
        <div class="perc-toggle on" id="pt-yolo" onclick="togglePerc('use_yolo')">
          <span>⚡ YOLO26</span>
          <span class="pt-dot"></span>
        </div>
        <div class="perc-toggle on" id="pt-gdino" onclick="togglePerc('use_gdino')">
          <span>🔍 GroundingDINO</span>
          <span class="pt-dot"></span>
        </div>
      </div>
      <div class="perc-hint">
        Distance priority: <span>LiDAR</span> → <span>Depth</span> → <span>Detections</span><br>
        Turn off sources to reduce compute. LiDAR always most accurate.
      </div>
    </div>

    <!-- YOLO Model Selector -->
    <div class="card">
      <h3>🤖 YOLO Model <span style="font-size:10px;color:#546E7A;font-weight:normal;">fast loop ~15fps · GDINO slow loop ~1fps</span></h3>
      <div class="yolo-select-wrap">
        <select class="yolo-select" id="yolo-select" onchange="switchYOLO(this.value)">
          <optgroup label="── YOLO26 (Latest 2025, NMS-free) ──">
            <option value="yolo26n" selected>YOLO26 Nano   — fastest, edge-optimised</option>
            <option value="yolo26s">YOLO26 Small  — fast,    small-object aware</option>
            <option value="yolo26m">YOLO26 Medium — balanced</option>
            <option value="yolo26l">YOLO26 Large  — accurate</option>
            <option value="yolo26x">YOLO26 XLarge — best accuracy</option>
          </optgroup>
          <optgroup label="── YOLO11 (2024) ──">
            <option value="yolo11n">YOLO11 Nano   — fastest,   ~5MB</option>
            <option value="yolo11s">YOLO11 Small  — fast,     ~19MB</option>
            <option value="yolo11m">YOLO11 Medium — balanced, ~39MB</option>
            <option value="yolo11l">YOLO11 Large  — accurate, ~49MB</option>
            <option value="yolo11x">YOLO11 XLarge — best,    ~109MB</option>
          </optgroup>
          <optgroup label="── YOLOv8 (Stable) ──">
            <option value="yolov8n">YOLOv8 Nano   — fastest,   ~6MB</option>
            <option value="yolov8s">YOLOv8 Small  — fast,     ~22MB</option>
            <option value="yolov8m">YOLOv8 Medium — balanced, ~52MB</option>
          </optgroup>
        </select>
      </div>
      <div class="yolo-status" id="yolo-status">Checking...</div>
    </div>

    <!-- Live Detections -->
    <div class="card">
      <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
        <h3 style="margin:0;">🎯 Detections</h3>
        <div style="display:flex; align-items:center; gap:6px;">
          <span style="font-size:10px; color:#546E7A;">
            <span class="src-dot-yolo"></span> YOLO
            &nbsp;<span class="src-dot-grounding"></span> GDINO
          </span>
          <button id="det-panel-btn" class="det-toggle-btn"
                  onclick="toggleDetections()"
                  style="width:auto; padding:4px 12px; font-size:12px;">
            <span class="dot"></span>
            <span id="det-panel-label">OFF</span>
          </button>
        </div>
      </div>
      <div id="det-status-row" style="font-size:10px; color:#546E7A; margin-bottom:6px; display:flex; gap:10px;">
        <span id="det-lidar-mode">—</span>
        <span id="det-perc-mode">—</span>
        <span id="det-obj-count" style="margin-left:auto; color:#A5D6A7; font-weight:bold;"></span>
      </div>
      <div class="det-scroll">
        <table class="det-table">
          <thead>
            <tr>
              <th style="width:14px;"></th>
              <th>Object</th>
              <th class="r">Dist</th>
              <th style="text-align:center">Sector</th>
              <th class="r">Conf</th>
            </tr>
          </thead>
          <tbody id="det-tbody">
            <tr><td colspan="5" class="det-loading">Perception pipeline loading…</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Live log -->
    <div class="card" style="flex:1">
      <h3>Live Log</h3>
      <div class="log-box" id="log-box"></div>
    </div>
  </div>
</div>

<script>
let busy = false;
let activeVLM = 'api';


async function switchVLMModel(model) {
  const st = document.getElementById('vlm-model-status');
  st.className = 'vlm-model-status';
  st.textContent = '⏳ Switching to ' + model + '...';
  try {
    const r = await fetch('/vlm/model/switch', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model})
    });
    const d = await r.json();
    if (d.error) {
      st.className = 'vlm-model-status err';
      st.textContent = '❌ ' + d.error;
    } else {
      st.className = 'vlm-model-status ok';
      st.textContent = '✅ ' + d.result;
    }
  } catch(e) {
    st.className = 'vlm-model-status err';
    st.textContent = 'Error: ' + e;
  }
}

async function updateVLMModelStatus() {
  try {
    const r = await fetch('/vlm/model/list');
    const d = await r.json();
    const sel = document.getElementById('vlm-model-select');
    const st  = document.getElementById('vlm-model-status');
    if (sel && d.active_model) {
      sel.value = d.active_model;
      // Only enable model selector when API backend is active
      sel.disabled = (d.active_backend !== 'api');
      sel.style.opacity = sel.disabled ? '0.4' : '1';
      st.className   = 'vlm-model-status ok';
      st.textContent = sel.disabled
        ? `Model switching requires API backend (current: ${d.active_backend})`
        : `✅ Active: ${d.active_model}`;
    }
  } catch {}
}


const _percState = {use_lidar:true, use_depth:true, use_yolo:true, use_gdino:true};
const _percIds   = {use_lidar:'pt-lidar', use_depth:'pt-depth',
                    use_yolo:'pt-yolo',   use_gdino:'pt-gdino'};

function _syncPercUI() {
  for (const [key, id] of Object.entries(_percIds)) {
    const el = document.getElementById(id);
    if (!el) continue;
    const on = _percState[key];
    el.className = 'perc-toggle ' + (on ? 'on' : 'off');
  }
}

async function togglePerc(key) {
  _percState[key] = !_percState[key];
  _syncPercUI();
  try {
    await fetch('/perc/config', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({[key]: _percState[key]})
    });
  } catch(e) {
    // Revert on failure
    _percState[key] = !_percState[key];
    _syncPercUI();
  }
}

async function updatePercConfig() {
  try {
    const r = await fetch('/perc/config');
    const d = await r.json();
    for (const key of ['use_lidar','use_depth','use_yolo','use_gdino']) {
      if (key in d) _percState[key] = d[key];
    }
    // Dim LiDAR toggle if hardware not present
    const lidarEl = document.getElementById('pt-lidar');
    if (lidarEl) {
      if (!d.has_lidar) {
        lidarEl.style.opacity = '0.45';
        lidarEl.title = 'No LiDAR data received yet';
      } else {
        lidarEl.style.opacity = '1';
        lidarEl.title = '';
      }
    }
    // Dim depth/yolo/gdino if perception not loaded yet
    const ready = d.perc_ready;
    for (const id of ['pt-depth','pt-yolo','pt-gdino']) {
      const el = document.getElementById(id);
      if (el) el.style.opacity = ready ? '1' : '0.45';
    }
    _syncPercUI();
  } catch {}
}

async function switchVLM(mode) {
  const r = await fetch('/vlm/switch', {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({mode})
  });
  const d = await r.json();
  activeVLM = d.active;
  updateVLMUI();
  document.getElementById('response').textContent = '⚙ ' + d.result;
}

function updateVLMUI() {
  ['api','local','remote'].forEach(m => {
    const btn = document.getElementById('btn-' + m);
    if (!btn) return;
    btn.className = (activeVLM === m) ? 'active-btn' : 'inactive-btn';
    if (m === 'remote' && activeVLM !== 'remote') {
      btn.style.color = '#FFE082';
      btn.style.borderColor = '#FFE082';
    } else if (m === 'remote') {
      btn.style.color = '';
      btn.style.borderColor = '';
    }
  });
  const badge = document.getElementById('top-vlm-badge');
  badge.textContent = 'VLM: ' + activeVLM.toUpperCase();
  badge.className = 'badge' +
    (activeVLM === 'local'  ? ' local'   :
     activeVLM === 'remote' ? ' loading' : '');
}

async function updateVLMInfo() {
  try {
    const r = await fetch('/vlm/status');
    const d = await r.json();
    activeVLM = d.active;
    const backends = Object.keys(d.backends);
    ['api','local','remote'].forEach(m => {
      const btn = document.getElementById('btn-' + m);
      if (btn) btn.style.display = backends.includes(m) ? '' : 'none';
    });
    updateVLMUI();
    const info = d.backends[d.active] || {};
    document.getElementById('vlm-info').innerHTML = Object.entries(info)
      .map(([k,v]) =>
        `<div><span class="key">${k}</span><span class="value">${v}</span></div>`)
      .join('');
  } catch {}
}

async function emergencyStop() {
  await fetch('/stop');
  document.getElementById('response').textContent = '⚡ Emergency stop sent!';
  busy = false;
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') emergencyStop();
});

async function cmd(text) {
  if (text === 'stop') { await emergencyStop(); return; }
  document.getElementById('cmd-input').value = text;
  await sendCmd();
}

async function sendCmd() {
  const text = document.getElementById('cmd-input').value.trim();
  if (!text) return;
  if (busy) { await fetch('/stop'); await new Promise(r=>setTimeout(r,600)); }
  busy = true;
  document.getElementById('response').textContent = '⏳ V-ROSA thinking...';
  try {
    const r = await fetch('/command', {
      method:'POST',
      headers:{'Content-Type':'application/x-www-form-urlencoded'},
      body:'text='+encodeURIComponent(text)
    });
    const d = await r.json();
    document.getElementById('response').textContent = d.response || d.error || 'No response';
  } catch(e) {
    document.getElementById('response').textContent = 'Error: '+e;
  }
  busy = false;
}

async function navTo() {
  const obj = document.getElementById('nav-input').value.trim();
  if (obj) await cmd('navigate to the ' + obj);
}
async function findObj() {
  const obj = document.getElementById('nav-input').value.trim();
  if (obj) await cmd('find the ' + obj + ' in your view');
}

async function updateStatus() {
  try {
    const r = await fetch('/status');
    const d = await r.json();
    document.getElementById('s-x').textContent     = d.x + 'm';
    document.getElementById('s-y').textContent     = d.y + 'm';
    document.getElementById('s-hdg').textContent   = d.heading + '°';
    document.getElementById('s-lidar').textContent = d.lidar_closest + 'm';
    document.getElementById('s-cam').textContent   = d.camera ? '✅ live' : '❌ none';
    document.getElementById('s-status').textContent= d.status;
    document.getElementById('top-status').textContent = d.status;
  } catch {}
}

async function updateLog() {
  try {
    const r = await fetch('/log_json');
    const d = await r.json();
    const box = document.getElementById('log-box');
    box.innerHTML = d.entries.slice(-40).reverse().map(e => {
      const tag = (e.match(/\\[([A-Z]+)\\]/g)||[]).pop()?.replace(/\\[|\\]/g,'') || 'INFO';
      return `<div class="log-${tag}">${e}</div>`;
    }).join('');
  } catch {}
}


let detOverlayOn = false;

async function toggleDetections() {
  try {
    const r = await fetch('/detections/toggle', {method:'POST'});
    const d = await r.json();
    detOverlayOn = d.show_dets;
    _syncDetUI();
  } catch(e) {
    console.error('toggle error', e);
  }
}

function _syncDetUI() {
  // Top-bar button
  const topBtn   = document.getElementById('det-top-btn');
  const topLabel = document.getElementById('det-top-label');
  // Panel button
  const panBtn   = document.getElementById('det-panel-btn');
  const panLabel = document.getElementById('det-panel-label');

  if (detOverlayOn) {
    topBtn.classList.add('on');
    topLabel.textContent = 'Detections ON';
    panBtn.classList.add('on');
    panLabel.textContent = 'ON';
  } else {
    topBtn.classList.remove('on');
    topLabel.textContent = 'Detections OFF';
    panBtn.classList.remove('on');
    panLabel.textContent = 'OFF';
  }
}


async function switchYOLO(key) {
  const statusEl = document.getElementById('yolo-status');
  statusEl.className = 'yolo-status loading';
  statusEl.textContent = '⏳ Loading ' + key + '...';
  try {
    const r = await fetch('/yolo/switch', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({model: key})
    });
    const d = await r.json();
    if (d.error) {
      statusEl.className = 'yolo-status';
      statusEl.textContent = '❌ ' + d.error;
    } else {
      statusEl.className = 'yolo-status loading';
      statusEl.textContent = '⏳ ' + d.result;
    }
  } catch(e) {
    statusEl.className = 'yolo-status';
    statusEl.textContent = 'Switch failed: ' + e;
  }
}

async function updateYOLOStatus() {
  try {
    const r = await fetch('/yolo/models');
    const d = await r.json();
    const sel    = document.getElementById('yolo-select');
    const status = document.getElementById('yolo-status');
    if (sel && d.current) {
      sel.value = d.current;
      status.className   = 'yolo-status ready';
      status.textContent = '✅ Active: ' + d.current;
    }
  } catch {}
}

async function updateDetections() {
  try {
    const r = await fetch('/detections');
    const d = await r.json();

    // Sync overlay state from server (in case of page reload)
    if (d.show_dets !== undefined && d.show_dets !== detOverlayOn) {
      detOverlayOn = d.show_dets;
      _syncDetUI();
    }

    // Status row
    const lidarMode = d.has_lidar ? '✅ LiDAR active' : '⚠ No LiDAR — Depth-only';
    let percStr;
    if (!d.perc_ready) {
      percStr = '⏳ Models loading…';
    } else {
      const ym = d.yolo_model || 'yolo26n';
      percStr = `🤖 ${ym.toUpperCase()}: ${d.n_yolo||0} · 🔍 GDINO: ${d.n_gdino||0}`;
    }
    document.getElementById('det-lidar-mode').textContent = lidarMode;
    document.getElementById('det-perc-mode').textContent  = percStr;

    const countEl = document.getElementById('det-obj-count');
    const topCount = document.getElementById('det-top-count');
    if (d.count > 0) {
      countEl.textContent  = d.count + ' object' + (d.count !== 1 ? 's' : '');
      topCount.textContent = ' · ' + d.count;
    } else {
      countEl.textContent  = '';
      topCount.textContent = '';
    }

    const tbody = document.getElementById('det-tbody');
    if (!d.detections || d.detections.length === 0) {
      const msg = d.perc_ready
        ? '<tr><td colspan="5" class="det-empty">Nothing detected right now</td></tr>'
        : '<tr><td colspan="5" class="det-loading">Perception pipeline loading…</td></tr>';
      tbody.innerHTML = msg;
      return;
    }
    tbody.innerHTML = d.detections.map(det => {
      const dotClass = det.source === 'grounding' ? 'src-dot-grounding' : 'src-dot-yolo';
      const dist  = det.dist >= 90 ? '—' : det.dist.toFixed(1) + 'm';
      const conf  = (det.conf * 100).toFixed(0) + '%';
      // Colour-code distance: red <1m, amber <2.5m, green otherwise
      const distColor = det.dist < 1.0 ? '#DC2626'
                       : det.dist < 2.5 ? '#D97706' : '#0369A1';
      return `<tr>
        <td><span class="${dotClass}"></span></td>
        <td class="det-label-val">${det.label}</td>
        <td style="color:${distColor}; font-weight:bold; text-align:right;">${dist}</td>
        <td class="det-sector-val">${det.sector}</td>
        <td class="det-conf-val">${conf}</td>
      </tr>`;
    }).join('');
  } catch {}
}


// ════════════════════════════════════════════════════════════
// MINI RVIZ — Top-down occupancy map with trajectory, yaw, zoom
// ════════════════════════════════════════════════════════════
const _rviz = {
  RANGE:    10.0,        // metres shown each side of robot (zoom level)
  RANGE_MIN: 3.0,
  RANGE_MAX: 25.0,
  canvas: null, ctx: null, last: null,
  trail:  [],            // [{x,y}] world-frame trajectory history
  trailMax: 120,
  animFrame: 0,          // for sweep animation
};

function rvizInit() {
  _rviz.canvas = document.getElementById('rviz-canvas');
  if (!_rviz.canvas) return;
  _rviz.ctx = _rviz.canvas.getContext('2d');
  rvizResize();
  window.addEventListener('resize', rvizResize);
  // Zoom with scroll wheel
  _rviz.canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 1.2 : 0.83;
    _rviz.RANGE = Math.min(_rviz.RANGE_MAX, Math.max(_rviz.RANGE_MIN, _rviz.RANGE * factor));
    if (_rviz.last) rvizDraw(_rviz.last);
  }, {passive: false});
}

function rvizResize() {
  const c = _rviz.canvas;
  if (!c) return;
  const w = c.clientWidth;
  c.width = w; c.height = w;
  if (_rviz.last) rvizDraw(_rviz.last);
}

// Robot frame → canvas pixel (robot always centred, yaw-rotated)
function rvizW2P(wx, wy, cx, cy, scale, cosYaw, sinYaw) {
  // Rotate world point by -yaw to get robot-relative coords
  const rx =  wx * cosYaw + wy * sinYaw;
  const ry = -wx * sinYaw + wy * cosYaw;
  return [cx + ry * scale, cy - rx * scale];
}

function rvizDraw(d) {
  const ctx = _rviz.ctx, c = _rviz.canvas;
  if (!ctx || !c) return;
  _rviz.animFrame = (_rviz.animFrame + 1) % 90;

  const W = c.width, H = c.height;
  const cx = W / 2, cy = H / 2;
  const scale = (W / 2) / _rviz.RANGE;

  const yawRad = (d.yaw || 0) * Math.PI / 180;
  const cosY = Math.cos(yawRad), sinY = Math.sin(yawRad);

  // ── Background: clean white paper with vignette ───────────
  ctx.fillStyle = '#FAFAF8';
  ctx.fillRect(0, 0, W, H);
  // Soft edge vignette
  const vig = ctx.createRadialGradient(cx, cy, W*0.35, cx, cy, W*0.72);
  vig.addColorStop(0, 'rgba(0,0,0,0)');
  vig.addColorStop(1, 'rgba(0,0,0,0.04)');
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, W, H);

  // ── Fine grid — subtle square paper ──────────────────────
  const gridM = 1.0;  // 1m grid
  const gridPx = gridM * scale;
  ctx.strokeStyle = '#EDE8E0'; ctx.lineWidth = 0.5;
  const offX = ((cx % gridPx) + gridPx) % gridPx;
  const offY = ((cy % gridPx) + gridPx) % gridPx;
  for (let gx = offX; gx < W; gx += gridPx) {
    ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke();
  }
  for (let gy = offY; gy < H; gy += gridPx) {
    ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke();
  }
  // Bold 5m grid
  ctx.strokeStyle = '#D6CEC4'; ctx.lineWidth = 0.8;
  const grid5 = 5 * scale;
  const off5X = ((cx % grid5) + grid5) % grid5;
  const off5Y = ((cy % grid5) + grid5) % grid5;
  for (let gx = off5X; gx < W; gx += grid5) {
    ctx.beginPath(); ctx.moveTo(gx,0); ctx.lineTo(gx,H); ctx.stroke();
  }
  for (let gy = off5Y; gy < H; gy += grid5) {
    ctx.beginPath(); ctx.moveTo(0,gy); ctx.lineTo(W,gy); ctx.stroke();
  }

  // ── Range rings ──────────────────────────────────────────
  ctx.setLineDash([3,5]);
  [[2,'rgba(220,38,38,0.25)'],[4,'rgba(3,105,161,0.20)'],[8,'rgba(120,113,108,0.15)']].forEach(([r,col]) => {
    const rPx = r * scale;
    if (rPx > 8) {
      ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(cx, cy, rPx, 0, Math.PI*2); ctx.stroke();
      ctx.fillStyle = '#A8A29E'; ctx.font = '8px Inter,monospace';
      ctx.fillText(r+'m', cx + rPx*0.707 + 2, cy - rPx*0.707 + 3);
    }
  });
  ctx.setLineDash([]);

  // ── LiDAR occupancy: render as occupancy-map style ────────
  const pts = d.lidar || [];

  // Group into angular bins for fill effect
  const BINS = 180;
  const binDists = new Float32Array(BINS).fill(99);
  pts.forEach(p => {
    // robot-relative bearing
    const bearing = Math.atan2(p.y, p.x);   // -π to π
    const binIdx  = Math.floor((bearing + Math.PI) / (2 * Math.PI / BINS)) % BINS;
    if (p.d < binDists[binIdx]) binDists[binIdx] = p.d;
  });

  // Fill occupied sector as a polygon (occupancy map effect)
  ctx.beginPath();
  let started = false;
  for (let i = 0; i <= BINS; i++) {
    const bIdx = i % BINS;
    const angle = (bIdx / BINS) * Math.PI * 2 - Math.PI;
    const dist  = Math.min(binDists[bIdx], _rviz.RANGE) * scale;
    // Robot frame → canvas (yaw-rotated)
    const worldX =  dist * Math.cos(angle);
    const worldY =  dist * Math.sin(angle);
    const px = cx - dist * Math.sin(angle);
    const py = cy - dist * Math.cos(angle);
    if (!started) { ctx.moveTo(px, py); started = true; }
    else          { ctx.lineTo(px, py); }
  }
  ctx.closePath();
  ctx.fillStyle   = 'rgba(3,105,161,0.06)';
  ctx.strokeStyle = 'rgba(3,105,161,0.20)';
  ctx.lineWidth   = 1;
  ctx.fill(); ctx.stroke();

  // Individual points coloured by distance
  pts.forEach(p => {
    // Rotate by yaw
    const px  = cx - p.y * scale;
    const py  = cy - p.x * scale;
    if (px < 0 || px > W || py < 0 || py > H) return;
    let col, sz;
    if      (p.d < 1.5) { col = '#DC2626'; sz = 2.5; }
    else if (p.d < 3.0) { col = '#EA580C'; sz = 2.0; }
    else if (p.d < 5.0) { col = '#0369A1'; sz = 1.5; }
    else if (p.d < 8.0) { col = '#60A5FA'; sz = 1.2; }
    else                 { col = '#C4B8A8'; sz = 1.0; }
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(px, py, sz, 0, Math.PI*2); ctx.fill();
  });

  // ── LiDAR sweep animation ─────────────────────────────────
  if (d.has_lidar && pts.length > 10) {
    const t = _rviz.animFrame / 90;
    const sweepA = t * Math.PI * 2 - Math.PI / 2;
    const g = ctx.createLinearGradient(
      cx, cy,
      cx + Math.cos(sweepA)*_rviz.RANGE*scale*0.9,
      cy + Math.sin(sweepA)*_rviz.RANGE*scale*0.9
    );
    g.addColorStop(0, 'rgba(3,105,161,0.30)');
    g.addColorStop(1, 'rgba(3,105,161,0)');
    ctx.strokeStyle = g; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx+Math.cos(sweepA)*_rviz.RANGE*scale*0.9,
               cy+Math.sin(sweepA)*_rviz.RANGE*scale*0.9);
    ctx.stroke();
  }

  // ── Path trail ────────────────────────────────────────────
  const trail = d.trail || [];
  const rx0 = d.x || 0, ry0 = d.y || 0;
  if (trail.length > 1) {
    for (let i = 1; i < trail.length; i++) {
      const a = trail[i-1], b = trail[i];
      const alpha = 0.12 + 0.65 * (i / trail.length);
      const dax = a.x-rx0, day = a.y-ry0;
      const dbx = b.x-rx0, dby = b.y-ry0;
      const rax =  dax*cosY + day*sinY,  ray = -dax*sinY + day*cosY;
      const rbx =  dbx*cosY + dby*sinY,  rby = -dbx*sinY + dby*cosY;
      const px1 = cx - ray*scale, py1 = cy - rax*scale;
      const px2 = cx - rby*scale, py2 = cy - rbx*scale;
      // Shadow
      ctx.strokeStyle = `rgba(3,105,161,${alpha*0.2})`;
      ctx.lineWidth = 5;
      ctx.beginPath(); ctx.moveTo(px1,py1); ctx.lineTo(px2,py2); ctx.stroke();
      // Line
      ctx.strokeStyle = `rgba(3,105,161,${alpha*0.85})`;
      ctx.lineWidth = 1.8;
      ctx.beginPath(); ctx.moveTo(px1,py1); ctx.lineTo(px2,py2); ctx.stroke();
    }
    // Start marker
    const fs  = trail[0];
    const wsx =  (fs.x-rx0)*cosY - (fs.y-ry0)*sinY;
    const wsy =  (fs.x-rx0)*sinY + (fs.y-ry0)*cosY;
    const psx = cx + wsy*scale, psy = cy - wsx*scale;
    ctx.fillStyle = 'rgba(3,105,161,0.25)';
    ctx.strokeStyle = '#0369A1'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.arc(psx, psy, 4, 0, Math.PI*2);
    ctx.fill(); ctx.stroke();
    // Total distance label
    if (trail.length > 4) {
      let dist = 0;
      for (let i=1;i<trail.length;i++)
        dist += Math.hypot(trail[i].x-trail[i-1].x, trail[i].y-trail[i-1].y);
      const mid  = trail[Math.floor(trail.length/2)];
      const wmx  = (mid.x-rx0)*cosY - (mid.y-ry0)*sinY;
      const wmy  = (mid.x-rx0)*sinY + (mid.y-ry0)*cosY;
      const pmx  = cx + wmy*scale, pmy = cy - wmx*scale;
      const lbl  = dist.toFixed(1)+'m';
      const ltw  = ctx.measureText(lbl).width + 8;
      ctx.fillStyle = 'rgba(255,255,255,0.90)';
      ctx.strokeStyle = '#0369A1'; ctx.lineWidth = 0.8;
      ctx.beginPath(); ctx.roundRect(pmx-ltw/2, pmy-8, ltw, 13, 3);
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#0369A1'; ctx.font = 'bold 8px Inter,monospace';
      ctx.fillText(lbl, pmx-ltw/2+4, pmy+2);
    }
  }

  // ── Safety zone ───────────────────────────────────────────
  const szR = 0.75 * scale;  // matches SAFE_DIST
  const szG = ctx.createRadialGradient(cx,cy,0, cx,cy,szR);
  szG.addColorStop(0,   'rgba(220,38,38,0.12)');
  szG.addColorStop(0.8, 'rgba(220,38,38,0.05)');
  szG.addColorStop(1,   'rgba(220,38,38,0)');
  ctx.fillStyle = szG;
  ctx.beginPath(); ctx.arc(cx,cy,szR,0,Math.PI*2); ctx.fill();

  // ── Detections ────────────────────────────────────────────
  const dets = d.dets || [];
  dets.forEach(det => {
    if (det.dist >= 90) return;
    const sA  = det.sector==='left' ? 0.4 : det.sector==='right' ? -0.4 : 0;
    // In robot frame, object is at (dist, ±dist*tan(sA)) ≈ (dist, dist*sA)
    const owx = det.dist;
    const owy = -det.dist * Math.sin(sA);
    // Rotate into canvas
    const dpx = cx - owy * scale;
    const dpy = cy - owx * scale;

    const isGdino = det.source === 'gdino';
    const col  = isGdino ? '#16A34A' : '#0369A1';
    const colA = isGdino ? 'rgba(22,163,74,' : 'rgba(3,105,161,';
    const dr   = Math.max(5, Math.min(14, scale*0.35));

    // Glow halo
    const glo = ctx.createRadialGradient(dpx,dpy,0, dpx,dpy,dr*2.5);
    glo.addColorStop(0, colA+'0.18)');
    glo.addColorStop(1, colA+'0)');
    ctx.fillStyle = glo;
    ctx.beginPath(); ctx.arc(dpx,dpy,dr*2.5,0,Math.PI*2); ctx.fill();
    // Circle
    ctx.strokeStyle = col; ctx.fillStyle = colA+'0.15)';
    ctx.lineWidth = 1.8;
    ctx.beginPath(); ctx.arc(dpx,dpy,dr,0,Math.PI*2);
    ctx.fill(); ctx.stroke();
    // Centre dot
    ctx.fillStyle = col;
    ctx.beginPath(); ctx.arc(dpx,dpy,2.5,0,Math.PI*2); ctx.fill();
    // Label pill
    const short = det.label.length > 9 ? det.label.slice(0,9) : det.label;
    const tag   = short + ' ' + (det.dist<90 ? det.dist.toFixed(1)+'m' : '?');
    ctx.font = 'bold 8px Inter,monospace';
    const tw = ctx.measureText(tag).width + 8;
    const lx = dpx + dr + 3, ly = dpy - 6;
    ctx.fillStyle = 'rgba(255,255,255,0.92)';
    ctx.strokeStyle = col; ctx.lineWidth = 0.7;
    ctx.beginPath(); ctx.roundRect(lx-2, ly-9, tw, 13, 3);
    ctx.fill(); ctx.stroke();
    ctx.fillStyle = col;
    ctx.fillText(tag, lx+2, ly+1);
  });

  // ── Navigation target dashed line ─────────────────────────
  if (d.status && d.status.includes('navigating') && dets.length > 0) {
    const tgt = dets[0];
    const sA  = tgt.sector==='left'?0.4:tgt.sector==='right'?-0.4:0;
    const twx = tgt.dist, twy = -tgt.dist*Math.sin(sA);
    const tcx = twx*cosY - twy*sinY, tcy = twx*sinY + twy*cosY;
    const tpx = cx+tcy*scale, tpy = cy-tcx*scale;
    ctx.setLineDash([5,4]);
    ctx.strokeStyle = '#16A34A'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(cx,cy); ctx.lineTo(tpx,tpy); ctx.stroke();
    ctx.setLineDash([]);
  }

  // ── Robot ─────────────────────────────────────────────────
  // Drop shadow
  ctx.fillStyle = 'rgba(0,0,0,0.10)';
  ctx.beginPath(); ctx.ellipse(cx+2, cy+2, 12, 9, 0, 0, Math.PI*2); ctx.fill();
  // Body
  const bg = ctx.createRadialGradient(cx-2,cy-3,1, cx,cy,11);
  bg.addColorStop(0, '#FFFFFF'); bg.addColorStop(1, '#DBEAFE');
  ctx.fillStyle = bg; ctx.strokeStyle = '#0369A1'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, 11, 0, Math.PI*2);
  ctx.fill(); ctx.stroke();
  // Forward arrow (always up = robot forward)
  ctx.fillStyle = '#0369A1';
  ctx.beginPath();
  ctx.moveTo(cx,    cy-9);
  ctx.lineTo(cx-4,  cy-1);
  ctx.lineTo(cx,    cy-3);
  ctx.lineTo(cx+4,  cy-1);
  ctx.closePath(); ctx.fill();
  // Heading badge
  ctx.fillStyle = '#78716C'; ctx.font = '7px Inter,monospace';
  ctx.fillText(Math.round(d.yaw||0)+'°', cx+13, cy+3);

  // ── Scale bar ─────────────────────────────────────────────
  const barM  = _rviz.RANGE <= 5 ? 1 : _rviz.RANGE <= 10 ? 2 : 5;
  const barPx = barM * scale;
  const bx = 8, by = H - 18;
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.fillRect(bx-2, by-10, barPx+4, 14);
  ctx.strokeStyle = '#78716C'; ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(bx, by); ctx.lineTo(bx+barPx, by);
  ctx.moveTo(bx, by-4); ctx.lineTo(bx, by+2);
  ctx.moveTo(bx+barPx, by-4); ctx.lineTo(bx+barPx, by+2);
  ctx.stroke();
  ctx.fillStyle = '#44403C'; ctx.font = 'bold 8px Inter,monospace';
  ctx.fillText(barM+'m', bx + barPx/2 - 6, by - 1);

  // ── HUD corner labels ─────────────────────────────────────
  ctx.fillStyle = d.has_lidar ? '#15803D' : '#B45309';
  ctx.font = 'bold 9px Inter,monospace';
  ctx.fillText(d.has_lidar ? '● LiDAR' : '○ Depth', 6, 13);

  const zStr = '±' + _rviz.RANGE.toFixed(0) + 'm';
  ctx.fillStyle = '#A8A29E'; ctx.font = '8px Inter,monospace';
  ctx.fillText(zStr, W - ctx.measureText(zStr).width - 5, 13);

  // Pose bottom-left
  ctx.fillStyle = '#78716C'; ctx.font = '8px Inter,monospace';
  ctx.fillText(`x=${(d.x||0).toFixed(2)} y=${(d.y||0).toFixed(2)}`, 6, H-5);

  // ── North compass ─────────────────────────────────────────
  const naX = W-14, naY = H-14;
  const nAng = -yawRad - Math.PI/2;
  ctx.save(); ctx.translate(naX, naY);
  // N label
  ctx.fillStyle = '#44403C'; ctx.font = 'bold 7px Inter,monospace';
  ctx.fillText('N', -3, -8);
  // Arrow
  ctx.rotate(nAng);
  ctx.fillStyle = '#DC2626';
  ctx.beginPath(); ctx.moveTo(0,-7); ctx.lineTo(-3,1); ctx.lineTo(0,-1); ctx.lineTo(3,1); ctx.closePath(); ctx.fill();
  ctx.fillStyle = '#C4B8A8';
  ctx.beginPath(); ctx.moveTo(0,-1); ctx.lineTo(-3,1); ctx.lineTo(0,7); ctx.lineTo(3,1); ctx.closePath(); ctx.fill();
  ctx.restore();
}


async function clearRvizTrail() {
  await fetch('/rviz_data/reset_trail', {method:'POST'});
}

async function updateRviz() {
  try {
    const r = await fetch('/rviz_data');
    const d = await r.json();
    // Trail now comes from server in d.trail
    _rviz.last = d;
    rvizDraw(d);
    const el = document.getElementById('rviz-pts-count');
    if (el) el.textContent =
      (d.lidar ? d.lidar.length + ' pts' : 'no lidar') +
      ' · ' + (d.dets ? d.dets.length + ' det' : '0 det') +
      ' · scroll=zoom';
  } catch {}
}


setInterval(updateDetections, 700);
setInterval(updateRviz, 100);
setInterval(updatePercConfig, 3000);
setInterval(updateVLMModelStatus, 4000);
setInterval(updateYOLOStatus, 4000);
setInterval(updateStatus, 800);
setInterval(updateLog, 1000);
setInterval(updateVLMInfo, 3000);
updateStatus(); updateLog(); updateVLMInfo(); updateDetections(); updateYOLOStatus(); updateVLMModelStatus(); updatePercConfig(); rvizInit(); updateRviz();
</script>
</body>
</html>
"""


@app.route("/view")
@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD)