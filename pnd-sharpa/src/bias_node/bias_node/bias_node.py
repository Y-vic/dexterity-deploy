#!/usr/bin/env python3
"""HTTP UI and ROS publisher for bias."""

from __future__ import annotations

import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from bias_node.body_bias import (
    body_display_name,
    complete_body_positions,
    default_bias_path,
    default_urdf_path,
    load_bias_joint_sets_file,
    load_joint_limits,
    normalize_joint_set_map,
    validate_within_limits,
    write_bias_joint_sets_file,
)
from bias_node.body_joints import (
    ADAM_COMMAND_JOINTS_19,
    UPPER_BODY_EDITABLE_GROUPS,
    UPPER_BODY_EDITABLE_JOINTS,
    canonical_body_name,
)


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bias</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #17202a;
      --muted: #5c6878;
      --accent: #1769aa;
      --accent-strong: #0f4c81;
      --danger: #a12828;
      --ok: #1e7b49;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 14px 18px;
    }
    .top {
      max-width: 1180px;
      margin: 0 auto;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }
    .actions {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .mode-switch {
      display: flex;
      align-items: center;
      gap: 0;
    }
    .mode-switch button {
      border-radius: 0;
      margin-left: -1px;
    }
    .mode-switch button:first-child {
      border-radius: 6px 0 0 6px;
      margin-left: 0;
    }
    .mode-switch button:last-child { border-radius: 0 6px 6px 0; }
    button {
      border: 1px solid var(--line);
      background: #ffffff;
      color: var(--text);
      border-radius: 6px;
      min-height: 34px;
      padding: 7px 12px;
      font-size: 14px;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
    }
    button.mode.active {
      border-color: var(--accent);
      background: #e8f2fb;
      color: var(--accent-strong);
      position: relative;
      z-index: 1;
    }
    button.danger {
      border-color: var(--danger);
      color: var(--danger);
    }
    button:hover { border-color: var(--accent-strong); }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 16px 18px 28px;
    }
    .status {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      min-width: 0;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .value {
      font-size: 14px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    section { margin-top: 14px; }
    h2 {
      font-size: 16px;
      margin: 0 0 8px;
      font-weight: 650;
    }
    .table-wrap {
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
      min-width: 820px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      font-size: 13px;
      text-align: left;
      vertical-align: middle;
    }
    th {
      color: var(--muted);
      font-weight: 600;
      background: #fbfcfe;
    }
    tr:last-child td { border-bottom: 0; }
    .joint { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .num { font-variant-numeric: tabular-nums; }
    input[type="number"] {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px 8px;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }
    input[type="number"]:focus {
      outline: 2px solid rgba(23, 105, 170, 0.18);
      border-color: var(--accent);
    }
    .message {
      min-height: 22px;
      margin-top: 10px;
      font-size: 13px;
      color: var(--muted);
    }
    .message.ok { color: var(--ok); }
    .message.error { color: var(--danger); }
    @media (max-width: 760px) {
      .top { align-items: flex-start; flex-direction: column; }
      .actions { justify-content: flex-start; }
      .status { grid-template-columns: 1fr; }
      main { padding: 12px; }
      header { padding: 12px; }
    }
  </style>
</head>
<body>
  <header>
    <div class="top">
      <h1>Bias Joints Set</h1>
      <div class="actions">
        <div class="mode-switch" aria-label="Edit mode">
          <button id="modeInit" class="mode">Bias Init</button>
          <button id="modeBias" class="mode active">Bias</button>
        </div>
        <button id="check">Check</button>
        <button id="reset">Reset Changes</button>
        <button id="save" class="primary">Save Bias</button>
      </div>
    </div>
  </header>
  <main>
    <div class="status">
      <div class="metric"><div class="label">Output Topic</div><div class="value" id="outputTopic">-</div></div>
      <div class="metric"><div class="label">Bias File</div><div class="value" id="biasPath">-</div></div>
      <div class="metric"><div class="label">Bias State</div><div class="value" id="biasState">-</div></div>
      <div class="metric"><div class="label">Current Age</div><div class="value" id="currentAge">-</div></div>
    </div>
    <div id="groups"></div>
    <div id="message" class="message"></div>
  </main>
  <script>
    const groupsEl = document.getElementById("groups");
    const messageEl = document.getElementById("message");
    const inputs = new Map();
    const applyTimers = new Map();
    let firstRender = true;
    let editMode = "bias";
    let lastState = null;
    const MODE_LABELS = {
      bias_init: "Bias Init",
      bias: "Bias"
    };
    function apiBaseFromPath() {
      const prefixes = ["/bias_joints", "/joint_bias"];
      for (const prefix of prefixes) {
        if (window.location.pathname === prefix || window.location.pathname.startsWith(prefix + "/")) {
          return prefix;
        }
      }
      return "";
    }

    const API_BASE = apiBaseFromPath();
    const RAD_TO_DEG = 180 / Math.PI;
    const DEG_TO_RAD = Math.PI / 180;

    function fmtRad(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(6);
    }

    function fmtDegFromRad(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return (Number(value) * RAD_TO_DEG).toFixed(2);
    }

    function degValue(value) {
      if (value === null || value === undefined || Number.isNaN(value)) return "";
      return (Number(value) * RAD_TO_DEG).toFixed(2);
    }

    function setMessage(text, type) {
      messageEl.textContent = text || "";
      messageEl.className = "message" + (type ? " " + type : "");
    }

    function modeLabel() {
      return MODE_LABELS[editMode] || "Bias";
    }

    function activeJointValue(joint) {
      return editMode === "bias_init" ? joint.bias_init : joint.bias;
    }

    function clearPendingEdits() {
      for (const timer of applyTimers.values()) {
        clearTimeout(timer);
      }
      applyTimers.clear();
      for (const input of inputs.values()) {
        input.dataset.dirty = "";
      }
    }

    function setEditMode(mode) {
      if (mode !== "bias_init" && mode !== "bias") return;
      if (mode === editMode) return;
      clearPendingEdits();
      editMode = mode;
      document.getElementById("modeInit").classList.toggle("active", editMode === "bias_init");
      document.getElementById("modeBias").classList.toggle("active", editMode === "bias");
      setMessage("");
      if (lastState) renderState(lastState);
    }

    async function requestJson(path, options) {
      const response = await fetch(API_BASE + path, options);
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function collectBiases() {
      const joints = {};
      for (const [name, input] of inputs.entries()) {
        const value = Number(input.value);
        if (!Number.isFinite(value)) {
          throw new Error(name + " has an invalid " + modeLabel());
        }
        joints[name] = value * DEG_TO_RAD;
      }
      return joints;
    }

    async function updateBiases(joints) {
      const data = await requestJson("/api/bias", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({mode: editMode, joints})
      });
      setMessage(modeLabel() + " applied", "ok");
      renderState(data.state);
    }

    function updateInputBias(name, input) {
      const value = Number(input.value);
      if (!Number.isFinite(value)) {
        setMessage(name + " has an invalid " + modeLabel(), "error");
        return;
      }
      updateBiases({[name]: value * DEG_TO_RAD})
        .then(() => {
          input.dataset.dirty = "";
          input.dataset.lastApplied = input.value;
        })
        .catch(err => setMessage(err.message, "error"));
    }

    function scheduleBiasUpdate(name, input) {
      if (applyTimers.has(name)) {
        clearTimeout(applyTimers.get(name));
      }
      const timer = setTimeout(() => {
        applyTimers.delete(name);
        updateInputBias(name, input);
      }, 250);
      applyTimers.set(name, timer);
    }

    async function saveBias() {
      const joints = collectBiases();
      const data = await requestJson("/api/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({mode: editMode, joints})
      });
      setMessage(modeLabel() + " saved", "ok");
      renderState(data.state);
    }

    async function resetChanges() {
      for (const timer of applyTimers.values()) {
        clearTimeout(timer);
      }
      applyTimers.clear();
      const data = await requestJson("/api/reset", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({mode: editMode})
      });
      for (const input of inputs.values()) {
        input.dataset.dirty = "";
      }
      setMessage(modeLabel() + " reset", "ok");
      renderState(data.state);
    }

    async function runCheck() {
      clearPendingEdits();
      const data = await requestJson("/api/check", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({})
      });
      setMessage("Check started", "ok");
      renderState(data.state);
    }

    function renderState(state) {
      lastState = state;
      document.getElementById("outputTopic").textContent = state.topics.output;
      document.getElementById("biasPath").textContent = state.bias_path;
      document.getElementById("save").textContent = "Save " + modeLabel();
      document.getElementById("biasState").textContent =
        state.publish_mode + ", active " + state.active_target.mode
        + " / " + state.active_target.phase
        + ", err " + (state.active_target.max_error_deg === null ? "-" : state.active_target.max_error_deg.toFixed(2))
        + " deg";
      const ages = [state.age_ms.robot_state, state.age_ms.command_state].filter(v => v !== null);
      document.getElementById("currentAge").textContent =
        ages.length === 0 ? "-" : Math.min(...ages) + " ms";

      if (firstRender) {
        groupsEl.innerHTML = "";
      }

      for (const group of state.groups) {
        let section = document.querySelector("section[data-group='" + group.name + "']");
        if (!section) {
          section = document.createElement("section");
          section.dataset.group = group.name;
          section.innerHTML = "<h2></h2><div class='table-wrap'><table><thead><tr>"
            + "<th style='width: 24%'>Joint</th>"
            + "<th style='width: 14%'>Current Rad</th>"
            + "<th style='width: 12%'>Current Deg</th>"
            + "<th style='width: 24%'>URDF Limit Deg</th>"
            + "<th class='target-header' style='width: 26%'>Bias Deg</th>"
            + "</tr></thead><tbody></tbody></table></div>";
          section.querySelector("h2").textContent = group.name;
          groupsEl.appendChild(section);
        }
        section.querySelector(".target-header").textContent = modeLabel() + " Deg";

        const tbody = section.querySelector("tbody");
        if (firstRender) tbody.innerHTML = "";
        for (const joint of group.joints) {
          let row = document.querySelector("tr[data-joint='" + joint.name + "']");
          if (!row) {
            row = document.createElement("tr");
            row.dataset.joint = joint.name;
            row.innerHTML = "<td class='joint'></td><td class='num current-rad'></td>"
              + "<td class='num current-deg'></td><td class='num limit'></td>"
              + "<td><input type='number' step='0.1'></td>";
            tbody.appendChild(row);
            row.querySelector(".joint").textContent = joint.display_name;
            const input = row.querySelector("input");
            input.min = degValue(joint.lower);
            input.max = degValue(joint.upper);
            input.value = degValue(activeJointValue(joint));
            input.addEventListener("input", () => {
              input.dataset.dirty = "1";
              scheduleBiasUpdate(joint.name, input);
            });
            input.addEventListener("change", () => {
              if (applyTimers.has(joint.name)) {
                clearTimeout(applyTimers.get(joint.name));
                applyTimers.delete(joint.name);
              }
              updateInputBias(joint.name, input);
            });
            input.addEventListener("keydown", (event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                if (applyTimers.has(joint.name)) {
                  clearTimeout(applyTimers.get(joint.name));
                  applyTimers.delete(joint.name);
                }
                updateInputBias(joint.name, input);
              }
            });
            inputs.set(joint.name, input);
          }
          row.querySelector(".current-rad").textContent = fmtRad(joint.current);
          row.querySelector(".current-deg").textContent = fmtDegFromRad(joint.current);
          row.querySelector(".limit").textContent =
            "[" + fmtDegFromRad(joint.lower) + ", " + fmtDegFromRad(joint.upper) + "]";
          const input = inputs.get(joint.name);
          if (input && document.activeElement !== input && input.dataset.dirty !== "1") {
            input.min = degValue(joint.lower);
            input.max = degValue(joint.upper);
            input.value = degValue(activeJointValue(joint));
          }
        }
      }
      firstRender = false;
    }

    async function refresh() {
      try {
        const state = await requestJson("/api/state");
        renderState(state);
      } catch (err) {
        setMessage(err.message, "error");
      }
    }

    document.getElementById("save").addEventListener("click", () => {
      saveBias().catch(err => setMessage(err.message, "error"));
    });
    document.getElementById("reset").addEventListener("click", () => {
      resetChanges().catch(err => setMessage(err.message, "error"));
    });
    document.getElementById("check").addEventListener("click", () => {
      runCheck().catch(err => setMessage(err.message, "error"));
    });
    document.getElementById("modeInit").addEventListener("click", () => setEditMode("bias_init"));
    document.getElementById("modeBias").addEventListener("click", () => setEditMode("bias"));

    refresh();
    setInterval(refresh, 500);
  </script>
</body>
</html>
"""


BIAS_INIT_MODE = "bias_init"
BIAS_MODE = "bias"
CONTROL_DAMPING_STATE = "damping"
T_INIT_CONTROL_STATE = "t_init"
STARTUP_SEQUENCE_BIAS_INIT_THEN_BIAS = "bias_init_then_bias"
STARTUP_SEQUENCE_BIAS = "bias"
STARTUP_SEQUENCE_MODES = {
    STARTUP_SEQUENCE_BIAS_INIT_THEN_BIAS,
    STARTUP_SEQUENCE_BIAS,
}
SequenceStep = tuple[str, str, dict[str, float]]


def startup_sequence_steps(
    mode: str,
    bias_init_positions: dict[str, float],
    bias_positions: dict[str, float],
) -> list[SequenceStep]:
    if mode == STARTUP_SEQUENCE_BIAS:
        return [("bias", BIAS_MODE, dict(bias_positions))]
    if mode == STARTUP_SEQUENCE_BIAS_INIT_THEN_BIAS:
        return [
            ("startup", BIAS_INIT_MODE, dict(bias_init_positions)),
            ("bias", BIAS_MODE, dict(bias_positions)),
        ]
    raise ValueError(
        "startup_sequence_mode must be bias_init_then_bias or bias"
    )


class _BiasServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, node: "DeployBiasJointSetNode") -> None:
        self.node = node
        super().__init__(address, handler)


class _BiasHandler(BaseHTTPRequestHandler):
    server: _BiasServer

    def do_OPTIONS(self) -> None:
        self._send_bytes(b"", content_type="text/plain")

    def do_GET(self) -> None:
        path = self._normalized_path()
        if path == "/":
            self._send_bytes(HTML.encode("utf-8"), content_type="text/html; charset=utf-8")
            return
        if path == "/api/state":
            self._send_json(self.server.node.state_payload())
            return
        self.send_error(404, "not found")

    def do_POST(self) -> None:
        path = self._normalized_path()
        try:
            payload = self._read_json()
            if path == "/api/bias":
                state = self.server.node.update_biases(payload, save=False)
                self._send_json({"ok": True, "state": state})
                return
            if path == "/api/save":
                state = self.server.node.update_biases(payload, save=True)
                self._send_json({"ok": True, "state": state})
                return
            if path == "/api/reset":
                state = self.server.node.reset_biases(payload)
                self._send_json({"ok": True, "state": state})
                return
            if path == "/api/check":
                state = self.server.node.start_check_sequence()
                self._send_json({"ok": True, "state": state})
                return
            self.send_error(404, "not found")
        except Exception as exc:  # noqa: BLE001 - report UI/API errors as JSON.
            self._send_json({"ok": False, "error": str(exc)}, status=400)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be a JSON object")
        return data

    def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
        self._send_bytes(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8"),
            status=status,
            content_type="application/json; charset=utf-8",
        )

    def _send_bytes(self, payload: bytes, *, status: int = 200, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.end_headers()
        if payload:
            self.wfile.write(payload)

    def _normalized_path(self) -> str:
        path = urlparse(self.path).path or "/"
        prefixes = []
        public_url_path = self.server.node.public_url_path.rstrip("/")
        if public_url_path:
            prefixes.append(public_url_path)
        prefixes.extend(["/bias_joints", "/joint_bias"])
        for prefix in dict.fromkeys(prefixes):
            if path == prefix or path == prefix + "/":
                return "/"
            if path.startswith(prefix + "/"):
                return path[len(prefix) :] or "/"
        return path

    def log_message(self, fmt: str, *args) -> None:
        self.server.node.get_logger().debug(fmt % args)


def transient_local_qos(depth: int = 1) -> QoSProfile:
    return QoSProfile(
        depth=depth,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def validate_complete_bias_joint_sets(
    bias_init: dict[str, float],
    bias: dict[str, float],
    *,
    path: str,
) -> None:
    missing_bias_init = [
        name for name in UPPER_BODY_EDITABLE_JOINTS if name not in bias_init
    ]
    missing_bias = [name for name in UPPER_BODY_EDITABLE_JOINTS if name not in bias]
    if missing_bias_init or missing_bias:
        raise ValueError(
            f"Bias file {path} is incomplete: "
            f"missing bias_init={missing_bias_init}, missing bias={missing_bias}"
        )


class DeployBiasJointSetNode(Node):
    def __init__(self) -> None:
        super().__init__("bias")

        self.declare_parameter("bind_host", "10.10.20.127")
        self.declare_parameter("bind_port", 18080)
        self.declare_parameter("bias_path", default_bias_path())
        self.declare_parameter("urdf_path", default_urdf_path())
        self.declare_parameter("output_topic", "/adam_bias_command_joint_states")
        self.declare_parameter("robot_state_topic", "/adam_physical_joint_states")
        self.declare_parameter("command_state_topic", "/adam_bias_command_joint_states")
        self.declare_parameter("control_status_topic", "/control_status")
        self.declare_parameter("status_topic", "/bias/status")
        self.declare_parameter("public_url_path", "")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("startup_publish_delay_s", 0.25)
        self.declare_parameter("startup_on_t_init", True)
        self.declare_parameter(
            "startup_sequence_mode",
            STARTUP_SEQUENCE_BIAS_INIT_THEN_BIAS,
        )
        self.declare_parameter("arrival_tolerance_deg", 8.0)
        self.declare_parameter("robot_state_timeout_s", 0.5)
        self.declare_parameter("sequence_step_timeout_s", 3.0)
        self.declare_parameter("interpolation_threshold_deg", 10.0)
        self.declare_parameter("interpolation_duration_s", 1.0)

        self.bind_host = str(self.get_parameter("bind_host").value)
        self.bind_port = int(self.get_parameter("bind_port").value)
        self.bias_path = str(self.get_parameter("bias_path").value)
        self.urdf_path = str(self.get_parameter("urdf_path").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.robot_state_topic = str(self.get_parameter("robot_state_topic").value)
        self.command_state_topic = str(self.get_parameter("command_state_topic").value)
        self.control_status_topic = str(self.get_parameter("control_status_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.public_url_path = str(self.get_parameter("public_url_path").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.startup_publish_delay_s = float(
            self.get_parameter("startup_publish_delay_s").value
        )
        self.startup_on_t_init = bool(self.get_parameter("startup_on_t_init").value)
        self.startup_sequence_mode = str(
            self.get_parameter("startup_sequence_mode").value
        ).strip().lower()
        arrival_tolerance_deg = float(
            self.get_parameter("arrival_tolerance_deg").value
        )
        self.arrival_tolerance_rad = math.radians(arrival_tolerance_deg)
        self.robot_state_timeout_s = float(
            self.get_parameter("robot_state_timeout_s").value
        )
        self.sequence_step_timeout_s = float(
            self.get_parameter("sequence_step_timeout_s").value
        )
        interpolation_threshold_deg = float(
            self.get_parameter("interpolation_threshold_deg").value
        )
        self.interpolation_threshold_rad = math.radians(interpolation_threshold_deg)
        self.interpolation_duration_s = float(
            self.get_parameter("interpolation_duration_s").value
        )
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.startup_publish_delay_s < 0.0:
            raise ValueError("startup_publish_delay_s must be non-negative")
        if self.startup_sequence_mode not in STARTUP_SEQUENCE_MODES:
            raise ValueError(
                "startup_sequence_mode must be bias_init_then_bias or bias"
            )
        if arrival_tolerance_deg < 0.0:
            raise ValueError("arrival_tolerance_deg must be non-negative")
        if self.robot_state_timeout_s <= 0.0:
            raise ValueError("robot_state_timeout_s must be positive")
        if self.sequence_step_timeout_s < 0.0:
            raise ValueError("sequence_step_timeout_s must be non-negative")
        if interpolation_threshold_deg < 0.0:
            raise ValueError("interpolation_threshold_deg must be non-negative")
        if self.interpolation_duration_s <= 0.0:
            raise ValueError("interpolation_duration_s must be positive")

        self.joint_limits = load_joint_limits(self.urdf_path)
        missing_limits = [name for name in UPPER_BODY_EDITABLE_JOINTS if name not in self.joint_limits]
        if missing_limits:
            raise ValueError(
                f"Missing URDF limits for joints from {self.urdf_path}: {missing_limits}"
            )

        bias_path = os.path.expanduser(self.bias_path)
        if not os.path.exists(bias_path):
            raise FileNotFoundError(f"Bias file is required: {bias_path}")
        try:
            saved_bias_init, saved_bias = load_bias_joint_sets_file(self.bias_path)
            validate_complete_bias_joint_sets(
                saved_bias_init,
                saved_bias,
                path=bias_path,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load required bias file {bias_path}: {exc}") from exc

        self.lock = threading.Lock()
        self.bias_init_positions = complete_body_positions(saved_bias_init)
        self.bias_positions = complete_body_positions(saved_bias)
        self.saved_bias_init_positions = dict(self.bias_init_positions)
        self.saved_positions = dict(self.bias_positions)
        self.reset_bias_init_positions = dict(self.bias_init_positions)
        self.reset_positions = dict(self.bias_positions)
        self.user_modified = False
        self.robot_positions: dict[str, float] = {}
        self.command_positions: dict[str, float] = {}
        self.robot_state_time: float | None = None
        self.command_state_time: float | None = None
        self.last_publish_time: float | None = None
        self.ramp_start_time: float | None = None
        self.ramp_end_time: float | None = None
        self.ramp_start_positions: dict[str, float] = {}
        self.ramp_target_positions: dict[str, float] = {}
        self.active_target_positions: dict[str, float] = dict(self.bias_positions)
        self.active_target_mode = BIAS_MODE
        self.active_action = "loaded_saved_bias"
        self.active_phase = "bias"
        self.control_state = CONTROL_DAMPING_STATE
        self.previous_control_state = CONTROL_DAMPING_STATE
        self.control_status_count = 0
        self.control_status_time: float | None = None
        self.last_startup_trigger = "waiting_for_t_init"
        self.pending_startup_trigger = ""
        self.startup_trigger_timer = None
        self.sequence_steps: list[SequenceStep] = []
        self.sequence_step_start_time: float | None = None
        self.sequence_step_timeout_reported = False
        self.sequence_active = False
        self.sequence_phase = "bias"
        self.last_interpolation_delta_rad = 0.0
        self.last_target_reached = False
        self.last_target_max_error_rad: float | None = None
        self.last_target_waiting_joints: list[str] = []
        self.last_sequence_advance_reason = ""
        self.published = 0
        self.saved = 0
        self.target_sequence = 0
        self.last_action = "loaded_saved_bias"
        self.last_target_mode = BIAS_MODE
        self.last_error = ""

        self.publisher = self.create_publisher(JointState, self.output_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)
        self.create_subscription(JointState, self.robot_state_topic, self._on_robot_state, 10)
        self.create_subscription(JointState, self.command_state_topic, self._on_command_state, 10)
        self.create_subscription(
            String,
            self.control_status_topic,
            self._on_control_status,
            transient_local_qos(),
        )
        self.create_timer(1.0 / self.publish_rate_hz, self._publish_active_target)
        self.create_timer(0.5, self._publish_status)

        self.server = _BiasServer((self.bind_host, self.bind_port), _BiasHandler, self)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            name="bias_http",
            daemon=True,
        )
        self.server_thread.start()
        self.get_logger().info(
            f"Bias joints set listening on {self.bind_host}:{self.bind_port}; "
            f"bias_path={self.bias_path}; target_topic={self.output_topic}; "
            f"control_status_topic={self.control_status_topic}; "
            f"startup_on_t_init={self.startup_on_t_init}; "
            f"startup_sequence_mode={self.startup_sequence_mode}; "
            f"startup_publish_delay_s={self.startup_publish_delay_s}; "
            f"arrival_tolerance_deg={arrival_tolerance_deg}; "
            f"robot_state_timeout_s={self.robot_state_timeout_s}; "
            f"sequence_step_timeout_s={self.sequence_step_timeout_s}; "
            f"interpolation_threshold_deg={interpolation_threshold_deg}; "
            f"interpolation_duration_s={self.interpolation_duration_s}"
        )

    def _on_robot_state(self, msg: JointState) -> None:
        positions = self._positions_from_msg(msg)
        with self.lock:
            self.robot_positions = positions
            self.robot_state_time = time.monotonic()

    def _on_command_state(self, msg: JointState) -> None:
        positions = self._positions_from_msg(msg)
        with self.lock:
            self.command_positions = positions
            self.command_state_time = time.monotonic()

    def _on_control_status(self, msg: String) -> None:
        state = (msg.data or "").strip()
        if not state:
            return
        should_trigger = False
        trigger = ""
        with self.lock:
            previous = self.control_state
            self.previous_control_state = previous
            self.control_state = state
            self.control_status_count += 1
            self.control_status_time = time.monotonic()
            if (
                self.startup_on_t_init
                and state == T_INIT_CONTROL_STATE
                and previous == CONTROL_DAMPING_STATE
            ):
                should_trigger = True
                trigger = f"{previous}_to_{state}"
                self.pending_startup_trigger = trigger

        if should_trigger:
            self._schedule_startup_sequence_from_status(trigger)

    def state_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        with self.lock:
            robot_positions = dict(self.robot_positions)
            command_positions = dict(self.command_positions)
            bias_init_positions = dict(self.bias_init_positions)
            bias_positions = dict(self.bias_positions)
            saved_bias_init_positions = dict(self.saved_bias_init_positions)
            saved_positions = dict(self.saved_positions)
            robot_state_time = self.robot_state_time
            command_state_time = self.command_state_time
            last_publish_time = self.last_publish_time
            ramp_end_time = self.ramp_end_time
            active_target_mode = self.active_target_mode
            active_action = self.active_action
            active_phase = self.active_phase
            active_target_positions = dict(self.active_target_positions)
            sequence_active = self.sequence_active
            sequence_phase = self.sequence_phase
            sequence_remaining = len(self.sequence_steps)
            sequence_step_start_time = self.sequence_step_start_time
            sequence_step_timeout_reported = self.sequence_step_timeout_reported
            control_state = self.control_state
            previous_control_state = self.previous_control_state
            control_status_count = self.control_status_count
            control_status_time = self.control_status_time
            last_startup_trigger = self.last_startup_trigger
            pending_startup_trigger = self.pending_startup_trigger
            last_interpolation_delta_rad = self.last_interpolation_delta_rad
            last_target_reached = self.last_target_reached
            last_target_max_error_rad = self.last_target_max_error_rad
            last_target_waiting_joints = list(self.last_target_waiting_joints)
            last_sequence_advance_reason = self.last_sequence_advance_reason
            published = self.published
            saved = self.saved
            last_action = self.last_action
            last_error = self.last_error
            interpolation_remaining_ms = (
                max(0.0, round((ramp_end_time - now) * 1000.0, 1))
                if ramp_end_time is not None and ramp_end_time > now
                else 0.0
            )

        groups = []
        for group_name, joints in UPPER_BODY_EDITABLE_GROUPS:
            group_payload = {"name": group_name, "joints": []}
            for name in joints:
                current = robot_positions.get(
                    name,
                    command_positions.get(
                        name,
                        active_target_positions.get(name, bias_positions.get(name, 0.0)),
                    ),
                )
                lower, upper = self.joint_limits[name]
                group_payload["joints"].append(
                    {
                        "name": name,
                        "display_name": body_display_name(name),
                        "current": current,
                        "bias_init": bias_init_positions.get(name, 0.0),
                        "bias": bias_positions.get(name, 0.0),
                        "saved_bias_init": saved_bias_init_positions.get(name),
                        "saved": saved_positions.get(name),
                        "lower": lower,
                        "upper": upper,
                    }
                )
            groups.append(group_payload)

        return {
            "ok": True,
            "node": "bias",
            "publish_rate_hz": self.publish_rate_hz,
            "arrival_tolerance_deg": math.degrees(self.arrival_tolerance_rad),
            "robot_state_timeout_s": self.robot_state_timeout_s,
            "sequence_step_timeout_s": self.sequence_step_timeout_s,
            "interpolation_threshold_deg": math.degrees(self.interpolation_threshold_rad),
            "interpolation_duration_s": self.interpolation_duration_s,
            "publish_mode": "adam_bias_command_joint_states",
            "sequence_active": sequence_active,
            "sequence_phase": sequence_phase,
            "sequence_remaining": sequence_remaining,
            "sequence_step_elapsed_s": (
                None
                if sequence_step_start_time is None or not sequence_active
                else round(max(0.0, now - sequence_step_start_time), 3)
            ),
            "sequence_step_timeout_reported": sequence_step_timeout_reported,
            "last_sequence_advance_reason": last_sequence_advance_reason,
            "active_target": {
                "mode": active_target_mode,
                "action": active_action,
                "phase": active_phase,
                "reached": last_target_reached,
                "max_error_deg": (
                    None
                    if last_target_max_error_rad is None
                    else math.degrees(last_target_max_error_rad)
                ),
                "waiting_joints": last_target_waiting_joints,
            },
            "interpolation_active": interpolation_remaining_ms > 0.0,
            "interpolation_remaining_ms": interpolation_remaining_ms,
            "last_interpolation_delta_deg": math.degrees(last_interpolation_delta_rad),
            "bias_path": self.bias_path,
            "urdf_path": self.urdf_path,
            "groups": groups,
            "topics": {
                "output": self.output_topic,
                "target": self.output_topic,
                "robot_state": self.robot_state_topic,
                "command_state": self.command_state_topic,
                "control_status": self.control_status_topic,
            },
            "control": {
                "state": control_state,
                "previous_state": previous_control_state,
                "startup_on_t_init": self.startup_on_t_init,
                "startup_sequence_mode": self.startup_sequence_mode,
                "t_init_state": T_INIT_CONTROL_STATE,
                "status_count": control_status_count,
                "status_age_ms": self._age_ms(control_status_time, now),
                "last_startup_trigger": last_startup_trigger,
                "pending_startup_trigger": pending_startup_trigger,
            },
            "counts": {"published": published, "saved": saved},
            "age_ms": {
                "robot_state": self._age_ms(robot_state_time, now),
                "command_state": self._age_ms(command_state_time, now),
                "last_publish": self._age_ms(last_publish_time, now),
            },
            "last_action": last_action,
            "last_error": last_error,
        }

    def update_biases(self, payload: dict[str, Any], *, save: bool) -> dict[str, Any]:
        mode = self._mode_from_payload(payload)
        updates = self._updates_from_payload(payload)
        validate_within_limits(updates, self.joint_limits)

        self._cancel_startup_timers()
        with self.lock:
            target_positions = (
                self.bias_init_positions if mode == "bias_init" else self.bias_positions
            )
            target_positions.update(updates)
            full = complete_body_positions(target_positions)
            if mode == "bias_init":
                self.bias_init_positions = full
            else:
                self.bias_positions = full
            if save:
                saved_bias_init, saved_bias = load_bias_joint_sets_file(self.bias_path)
                saved_bias_init_positions = complete_body_positions(saved_bias_init)
                saved_bias_positions = complete_body_positions(saved_bias)
                if mode == "bias_init":
                    file_bias_init = self.bias_init_positions
                    file_bias = saved_bias_positions
                else:
                    file_bias_init = saved_bias_init_positions
                    file_bias = self.bias_positions
                write_bias_joint_sets_file(
                    self.bias_path,
                    file_bias_init,
                    file_bias,
                    source="bias",
                )
                self.saved_bias_init_positions = dict(file_bias_init)
                self.saved_positions = dict(file_bias)
                self.reset_bias_init_positions = dict(file_bias_init)
                self.reset_positions = dict(file_bias)
                self.saved += 1
                self.user_modified = False
                self.last_action = f"{mode}_saved"
            elif updates:
                self.user_modified = True
                self.last_action = f"{mode}_updated"
            else:
                self.last_action = f"{mode}_unchanged"
            self.last_target_mode = mode
            self.sequence_active = False
            self.sequence_steps = []
            self.sequence_phase = "startup" if mode == BIAS_INIT_MODE else "bias"
            self.last_error = ""

        self._set_active_target(full, mode=mode)
        return self.state_payload()

    def reset_biases(self, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        mode = self._mode_from_payload(payload or {})
        saved_bias_init, saved_bias = load_bias_joint_sets_file(self.bias_path)
        self._cancel_startup_timers()
        with self.lock:
            self.saved_bias_init_positions = complete_body_positions(saved_bias_init)
            self.saved_positions = complete_body_positions(saved_bias)
            self.reset_bias_init_positions = dict(self.saved_bias_init_positions)
            self.reset_positions = dict(self.saved_positions)
            reset_source = (
                self.reset_bias_init_positions if mode == "bias_init" else self.reset_positions
            )
            full = complete_body_positions(reset_source)
            if mode == "bias_init":
                self.bias_init_positions = dict(full)
            else:
                self.bias_positions = dict(full)
            self.user_modified = False
            self.sequence_active = False
            self.sequence_steps = []
            self.sequence_phase = "startup" if mode == BIAS_INIT_MODE else "bias"
            self.last_action = f"{mode}_reset"
            self.last_target_mode = mode
            self.last_error = ""

        self._set_active_target(full, mode=mode)
        return self.state_payload()

    def _schedule_startup_sequence_from_status(self, trigger: str) -> None:
        timer = getattr(self, "startup_trigger_timer", None)
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            self.startup_trigger_timer = None
        delay = max(self.startup_publish_delay_s, 0.001)
        self.get_logger().info(
            f"Scheduling bias startup sequence after status transition {trigger}"
        )
        self.startup_trigger_timer = self.create_timer(
            delay,
            self._publish_startup_from_status,
        )

    def _publish_startup_from_status(self) -> None:
        timer = getattr(self, "startup_trigger_timer", None)
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            self.startup_trigger_timer = None
        with self.lock:
            if self.control_state != T_INIT_CONTROL_STATE:
                self.pending_startup_trigger = ""
                self.last_startup_trigger = (
                    f"skipped_status_{self.control_state}_before_startup"
                )
                self.last_action = self.last_startup_trigger
                return
            trigger = self.pending_startup_trigger or f"status_{self.control_state}"
            self.pending_startup_trigger = ""
            self.last_startup_trigger = trigger
            steps = startup_sequence_steps(
                self.startup_sequence_mode,
                self.bias_init_positions,
                self.bias_positions,
            )
        self.get_logger().info(
            f"Starting bias startup sequence from status trigger {trigger}: "
            f"mode={self.startup_sequence_mode}"
        )
        self._start_sequence(steps)

    def start_check_sequence(self) -> dict[str, Any]:
        self._cancel_startup_timers()
        with self.lock:
            steps = [
                ("check_to_bias", BIAS_MODE, dict(self.bias_positions)),
                ("check_to_bias_init", BIAS_INIT_MODE, dict(self.bias_init_positions)),
                ("check_to_zero", BIAS_INIT_MODE, complete_body_positions({})),
                (
                    "check_back_to_bias_init",
                    BIAS_INIT_MODE,
                    dict(self.bias_init_positions),
                ),
                ("check_back_to_bias", BIAS_MODE, dict(self.bias_positions)),
            ]
            self.last_error = ""
        self.get_logger().info(
            "Starting bias check sequence: bias -> bias_init -> zero -> bias_init -> bias"
        )
        self._start_sequence(steps)
        return self.state_payload()

    def _start_sequence(self, steps: list[SequenceStep]) -> None:
        if not steps:
            return
        label, mode, positions = steps[0]
        with self.lock:
            self.sequence_steps = steps[1:]
            self.sequence_active = True
            self.sequence_phase = label
            self.sequence_step_start_time = time.monotonic()
            self.sequence_step_timeout_reported = False
            self.last_action = label
            self.last_target_mode = mode
            self.last_error = ""
            self.last_sequence_advance_reason = ""
        self._set_active_target(positions, mode=mode, action=label, phase=label)

    def _cancel_startup_timers(self) -> None:
        timer = getattr(self, "startup_trigger_timer", None)
        if timer is not None:
            timer.cancel()
            self.destroy_timer(timer)
            self.startup_trigger_timer = None
        with self.lock:
            self.sequence_steps = []
            self.sequence_active = False
            self.sequence_step_timeout_reported = False
            self.sequence_phase = self.active_phase

    def _set_active_target(
        self,
        positions: dict[str, float],
        *,
        mode: str,
        action: str | None = None,
        phase: str | None = None,
        start_positions: dict[str, float] | None = None,
    ) -> None:
        if mode not in {BIAS_INIT_MODE, BIAS_MODE}:
            raise ValueError("mode must be bias_init or bias")
        now = time.monotonic()
        target = complete_body_positions(positions)
        forced_start = (
            complete_body_positions(start_positions)
            if start_positions is not None
            else None
        )
        with self.lock:
            action = action or self.last_action
            phase = phase or ("startup" if mode == BIAS_INIT_MODE else "bias")
            self.active_target_mode = mode
            self.active_action = action
            self.active_phase = phase
            self.sequence_phase = phase if not self.sequence_active else self.sequence_phase
            start = (
                forced_start
                if forced_start is not None
                else self._current_publish_start_locked(target)
            )
            self.active_target_positions = target
            max_delta = max(
                abs(target[name] - start[name])
                for name in UPPER_BODY_EDITABLE_JOINTS
            )
            self.last_interpolation_delta_rad = max_delta
            if max_delta > self.interpolation_threshold_rad:
                self.ramp_start_time = now
                self.ramp_end_time = now + self.interpolation_duration_s
                self.ramp_start_positions = start
                self.ramp_target_positions = target
                first_frame = self._interpolated_positions_locked(now)
            else:
                self._clear_ramp_locked()
                first_frame = target
            interpolation_active = (
                self.ramp_end_time is not None and self.ramp_end_time > now
            )
            self.last_target_reached = False
            self.last_target_max_error_rad = None
            self.last_target_waiting_joints = list(UPPER_BODY_EDITABLE_JOINTS)
        self._publish_positions(first_frame, mode, action, phase, interpolation_active)

    def _publish_active_target(self) -> None:
        now = time.monotonic()
        with self.lock:
            positions = self._interpolated_positions_locked(now)
            mode = self.active_target_mode
            action = self.active_action
            phase = self.active_phase
            interpolation_active = (
                self.ramp_end_time is not None and self.ramp_end_time > now
            )
        self._publish_positions(positions, mode, action, phase, interpolation_active)
        self._advance_sequence_if_reached()

    def _advance_sequence_if_reached(self) -> None:
        next_step: SequenceStep | None = None
        now = time.monotonic()
        timed_out = False
        hold_after_timeout = False
        timeout_detail = ""
        with self.lock:
            if not self.sequence_active:
                self._target_reached_locked(now)
                return
            reached = self._target_reached_locked(now)
            if not reached:
                timed_out = self._sequence_step_timed_out_locked(now)
            if not reached and not timed_out:
                return
            advance_reason = "target_reached"
            if timed_out:
                elapsed = (
                    0.0
                    if self.sequence_step_start_time is None
                    else now - self.sequence_step_start_time
                )
                advance_reason = f"timeout_after_{elapsed:.2f}s"
                timeout_detail = (
                    f"{self.sequence_phase} timed out after {elapsed:.2f}s; "
                    f"max_error_deg="
                    f"{math.degrees(self.last_target_max_error_rad or 0.0):.2f}; "
                    f"waiting={self.last_target_waiting_joints}"
                )
                hold_after_timeout = (
                    self.sequence_phase == "startup"
                    and self.active_target_mode == BIAS_INIT_MODE
                )
                if hold_after_timeout:
                    self.sequence_step_timeout_reported = True
                    self.last_sequence_advance_reason = (
                        f"timeout_holding_init_after_{elapsed:.2f}s"
                    )
                    self.last_error = timeout_detail
                    next_step = None
                else:
                    self.last_sequence_advance_reason = advance_reason
            else:
                self.last_sequence_advance_reason = advance_reason
            if hold_after_timeout:
                pass
            elif self.sequence_steps:
                next_step = self.sequence_steps.pop(0)
                label, mode, _positions = next_step
                self.sequence_phase = label
                self.sequence_step_start_time = now
                self.sequence_step_timeout_reported = False
                self.last_action = label
                self.last_target_mode = mode
                self.last_error = ""
            else:
                self.sequence_active = False
                self.sequence_phase = "done"
                self.last_action = "sequence_done"
                if self.active_target_mode == BIAS_MODE:
                    self.active_action = "bias"
                    self.active_phase = "bias"
                return
        if timed_out and timeout_detail:
            self.get_logger().warning(
                (
                    "Bias startup holding init after timeout: "
                    if hold_after_timeout
                    else "Bias sequence advancing without exact arrival: "
                )
                + timeout_detail
            )
        if hold_after_timeout:
            return
        if next_step is not None:
            label, mode, positions = next_step
            self._set_active_target(positions, mode=mode, action=label, phase=label)

    def _sequence_step_timed_out_locked(self, now: float) -> bool:
        if self.sequence_step_timeout_s <= 0.0:
            return False
        if self.sequence_step_timeout_reported:
            return False
        if self.sequence_step_start_time is None:
            return False
        if self.ramp_end_time is not None and self.ramp_end_time > now:
            return False
        return now - self.sequence_step_start_time >= self.sequence_step_timeout_s

    def _target_reached_locked(self, now: float) -> bool:
        if self.ramp_end_time is not None and self.ramp_end_time > now:
            self.last_target_reached = False
            self.last_target_max_error_rad = None
            self.last_target_waiting_joints = ["interpolation"]
            return False

        if (
            self.robot_state_time is None
            or now - self.robot_state_time > self.robot_state_timeout_s
        ):
            self.last_target_reached = False
            self.last_target_max_error_rad = None
            self.last_target_waiting_joints = ["robot_state_stale"]
            return False

        target = self.active_target_positions
        waiting: list[str] = []
        max_error = 0.0
        for name in UPPER_BODY_EDITABLE_JOINTS:
            if name not in self.robot_positions or name not in target:
                waiting.append(name)
                continue
            error = abs(float(self.robot_positions[name]) - float(target[name]))
            max_error = max(max_error, error)
            if error > self.arrival_tolerance_rad:
                waiting.append(name)

        self.last_target_max_error_rad = max_error
        self.last_target_waiting_joints = waiting[:20]
        self.last_target_reached = not waiting
        return self.last_target_reached

    def _current_publish_start_locked(
        self,
        fallback: dict[str, float],
    ) -> dict[str, float]:
        source = self.robot_positions or self.command_positions
        if not source:
            return dict(fallback)
        return {
            name: float(source.get(name, fallback[name]))
            for name in ADAM_COMMAND_JOINTS_19
        }

    def _interpolated_positions_locked(self, now: float) -> dict[str, float]:
        if (
            self.ramp_start_time is None
            or self.ramp_end_time is None
            or not self.ramp_start_positions
            or not self.ramp_target_positions
        ):
            return dict(self.active_target_positions or self.bias_positions)

        if now >= self.ramp_end_time:
            target = dict(self.ramp_target_positions)
            self._clear_ramp_locked()
            return target

        duration = max(self.ramp_end_time - self.ramp_start_time, 1e-6)
        alpha = min(1.0, max(0.0, (now - self.ramp_start_time) / duration))
        return {
            name: self.ramp_start_positions[name]
            + (self.ramp_target_positions[name] - self.ramp_start_positions[name]) * alpha
            for name in ADAM_COMMAND_JOINTS_19
        }

    def _clear_ramp_locked(self) -> None:
        self.ramp_start_time = None
        self.ramp_end_time = None
        self.ramp_start_positions = {}
        self.ramp_target_positions = {}

    def _publish_positions(
        self,
        positions: dict[str, float],
        mode: str,
        action: str,
        phase: str,
        interpolation_active: bool,
    ) -> None:
        full = complete_body_positions(positions)
        now = time.monotonic()
        with self.lock:
            self.target_sequence += 1
            sequence = self.target_sequence
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f"adam_bias_command:{mode}:{action}:{phase}"
        msg.name = list(ADAM_COMMAND_JOINTS_19)
        msg.position = [float(full[name]) for name in ADAM_COMMAND_JOINTS_19]
        msg.velocity = [0.0] * len(msg.name)
        msg.effort = [0.0] * len(msg.name)
        self.publisher.publish(msg)
        with self.lock:
            self.published += 1
            self.command_positions = dict(full)
            self.command_state_time = now
            self.last_publish_time = self.command_state_time

    def _updates_from_payload(self, payload: dict[str, Any]) -> dict[str, float]:
        if "name" in payload and "position" in payload:
            raw = {str(payload["name"]): payload["position"]}
        else:
            raw = payload
        updates = normalize_joint_set_map(raw)
        if not updates:
            return {}
        forbidden = [name for name in updates if name not in UPPER_BODY_EDITABLE_JOINTS]
        if forbidden:
            raise ValueError(f"bias joints set only accepts upper-body joints: {forbidden}")
        return updates

    @staticmethod
    def _mode_from_payload(payload: dict[str, Any]) -> str:
        if "mode" not in payload:
            raise ValueError("mode is required and must be bias_init or bias")
        mode = str(payload["mode"])
        if mode in {"init", BIAS_INIT_MODE}:
            return BIAS_INIT_MODE
        if mode == BIAS_MODE:
            return BIAS_MODE
        raise ValueError("mode must be bias_init or bias")

    def _publish_status(self) -> None:
        state = self.state_payload()
        msg = String()
        msg.data = json.dumps(
            {
                "node": state["node"],
                "publish_mode": state["publish_mode"],
                "publish_rate_hz": state["publish_rate_hz"],
                "arrival_tolerance_deg": state["arrival_tolerance_deg"],
                "robot_state_timeout_s": state["robot_state_timeout_s"],
                "sequence_step_timeout_s": state["sequence_step_timeout_s"],
                "interpolation_threshold_deg": state["interpolation_threshold_deg"],
                "interpolation_duration_s": state["interpolation_duration_s"],
                "sequence_active": state["sequence_active"],
                "sequence_phase": state["sequence_phase"],
                "sequence_remaining": state["sequence_remaining"],
                "sequence_step_elapsed_s": state["sequence_step_elapsed_s"],
                "last_sequence_advance_reason": state["last_sequence_advance_reason"],
                "active_target": state["active_target"],
                "interpolation_active": state["interpolation_active"],
                "interpolation_remaining_ms": state["interpolation_remaining_ms"],
                "last_interpolation_delta_deg": state["last_interpolation_delta_deg"],
                "bias_path": state["bias_path"],
                "control": state["control"],
                "topics": state["topics"],
                "counts": state["counts"],
                "age_ms": state["age_ms"],
                "last_action": state["last_action"],
                "last_error": state["last_error"],
                "url": self.public_url_path or f"http://{self.bind_host}:{self.bind_port}/",
            },
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        self.status_pub.publish(msg)

    def destroy_node(self) -> bool:
        self.server.shutdown()
        self.server.server_close()
        return super().destroy_node()

    @staticmethod
    def _positions_from_msg(msg: JointState) -> dict[str, float]:
        positions: dict[str, float] = {}
        for idx, name in enumerate(msg.name):
            canonical = canonical_body_name(name)
            if canonical not in ADAM_COMMAND_JOINTS_19:
                continue
            if idx >= len(msg.position):
                raise ValueError(f"JointState position is missing for {canonical}")
            try:
                value = float(msg.position[idx])
            except (TypeError, ValueError):
                raise ValueError(
                    f"non-finite joint value for {canonical}: {msg.position[idx]!r}"
                ) from None
            if not math.isfinite(value):
                raise ValueError(
                    f"non-finite joint value for {canonical}: {msg.position[idx]!r}"
                )
            positions[canonical] = value
        return positions

    @staticmethod
    def _age_ms(timestamp: float | None, now: float) -> float | None:
        if timestamp is None:
            return None
        return round((now - timestamp) * 1000.0, 1)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DeployBiasJointSetNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
