"""Timestamped ten-finger measurements and the 2-by-6 demo video panel."""

from __future__ import annotations

from collections import deque
import io
import math
import threading

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FINGERS = ("thumb", "index", "middle", "ring", "pinky")
JOINT_GROUPS = ((0, 5), (5, 9), (9, 13), (13, 17), (17, 22))
SIDES = ("left", "right")
TORQUE_WINDOW_NS = 8_000_000_000
PANEL_COLUMNS = 2
PANEL_ROWS = 6


def _font(size, *, bold=False):
    names = ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf") if bold else (
        "DejaVuSans.ttf", "DejaVuSans-Bold.ttf")
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def finger_square_sums(q):
    values = np.asarray(q, dtype=float)
    if values.shape != (44,) or not np.isfinite(values).all():
        raise ValueError("expected 44 finite measured joint angles in radians")
    return [[float(np.dot(values[offset + a:offset + b],
                          values[offset + a:offset + b]))
             for a, b in JOINT_GROUPS] for offset in (0, 22)]


class TorqueHistory:
    """Bounded samples used by the two hand torque plots."""

    def __init__(self, window_ns=TORQUE_WINDOW_NS, maxlen=1024):
        self.window_ns = int(window_ns)
        self.rows = deque(maxlen=maxlen)

    def update(self, metrics):
        stamp = int(metrics.get("timestamp_ns", 0))
        if stamp <= 0 or (self.rows and stamp <= self.rows[-1][0]):
            return
        row = {}
        for side in SIDES:
            values = [finger.get("joint_torque_square_sum_nm2")
                      for finger in metrics.get("fingers", []) if finger.get("side") == side]
            array = np.asarray(values, dtype=float) if None not in values else np.empty(0)
            row[side] = array.copy() if array.shape == (5,) and np.isfinite(array).all() else None
        self.rows.append((stamp, row))
        cutoff = stamp - self.window_ns
        while self.rows and self.rows[0][0] < cutoff:
            self.rows.popleft()

    def series(self, side):
        return [(stamp, row[side]) for stamp, row in self.rows if row[side] is not None]


class SensorHistory:
    def __init__(self, max_age_ns=250_000_000):
        self.max_age_ns = max_age_ns
        self._lock = threading.Lock()
        self._history = {key: deque(maxlen=64) for key in ("hand", "tau", "wrench", "deform")}

    def update(self, key, msg, received_ns):
        if key in self._history:
            with self._lock:
                self._history[key].append((received_ns, msg))

    def snapshot(self, timestamp_ns):
        with self._lock:
            selected = {key: next(((stamp, msg) for stamp, msg in reversed(rows)
                                   if stamp <= timestamp_ns), (0, None))
                        for key, rows in self._history.items()}
        result = {"timestamp_ns": timestamp_ns, "fingers": [], "sources": {}}
        for key, (stamp, msg) in selected.items():
            source_stamp = int(getattr(msg, "timestamp_ns", 0))
            fresh = bool(stamp and source_stamp > 0
                         and 0 <= timestamp_ns - stamp <= self.max_age_ns
                         and 0 <= timestamp_ns - source_stamp <= self.max_age_ns)
            result["sources"][key] = {"timestamp_ns": source_stamp,
                                      "received_ns": stamp, "fresh": fresh}
        hand, wrench, deform = (selected[key][1] for key in ("hand", "wrench", "deform"))
        tau = selected["tau"][1]
        try:
            sums = finger_square_sums(hand.q)
        except (AttributeError, ValueError, TypeError):
            sums = [[None] * 5 for _ in SIDES]
        planes = []
        for side_index, side in enumerate(SIDES):
            torques = np.asarray(getattr(tau, side, []), dtype=float)
            torque_ok = (result["sources"]["tau"]["fresh"]
                         and bool(getattr(tau, f"{side}_valid", False))
                         and torques.shape == (22,) and np.isfinite(torques).all())
            for index, finger in enumerate(FINGERS):
                # Wrench/deformation use pinky -> thumb; joint states use thumb -> pinky.
                tactile_index = 4 - index
                hand_ok = result["sources"]["hand"]["fresh"] and bool(getattr(hand, f"{side}_valid", False))
                force = None
                valid = list(getattr(wrench, f"{side}_valid", []))
                values = np.asarray(getattr(wrench, side, []), dtype=float)
                if (result["sources"]["wrench"]["fresh"] and len(valid) == 5
                        and valid[tactile_index] and values.shape == (30,)):
                    vector = values.reshape(5, 6)[tactile_index]
                    if np.isfinite(vector).all():
                        force = vector.tolist()
                plane = None
                valid = list(getattr(deform, f"{side}_valid", []))
                if result["sources"]["deform"]["fresh"] and len(valid) == 5 and valid[tactile_index]:
                    values = np.asarray(getattr(deform, side, []), dtype=np.uint8)
                    if values.size == 5 * 240 * 240:
                        plane = values.reshape(5, 240, 240)[tactile_index].copy()
                planes.append(plane)
                result["fingers"].append({
                    "side": side, "finger": finger,
                    "joint_square_sum_rad2": sums[side_index][index] if hand_ok else None,
                    "joint_torque_square_sum_nm2": float(np.dot(
                        torques[JOINT_GROUPS[index][0]:JOINT_GROUPS[index][1]],
                        torques[JOINT_GROUPS[index][0]:JOINT_GROUPS[index][1]],
                    )) if torque_ok else None,
                    "wrench": force,
                    "force_norm_n": float(np.linalg.norm(force[:3])) if force is not None else None,
                    "deformation_valid": plane is not None,
                })
        return result, planes


def _draw_tactile_tile(image, box, finger, plane, fonts):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    draw.rectangle(box, fill=(2, 4, 6), outline=(219, 225, 229), width=2)
    inset = max(3, round((x1 - x0) * 0.025))
    if plane is not None:
        tactile = Image.fromarray(plane).convert("L").convert("RGB")
        image.paste(tactile.resize((x1 - x0 - inset * 2, y1 - y0 - inset * 2)),
                    (x0 + inset, y0 + inset))
    else:
        draw.line((x0 + inset * 4, y0 + inset * 4,
                   x1 - inset * 4, y1 - inset * 4), fill=(76, 84, 91), width=2)
        draw.line((x1 - inset * 4, y0 + inset * 4,
                   x0 + inset * 4, y1 - inset * 4), fill=(76, 84, 91), width=2)
        label = "NO TACTILE"
        bounds = draw.textbbox((0, 0), label, font=fonts["small"])
        draw.text(((x0 + x1 - (bounds[2] - bounds[0])) / 2,
                   (y0 + y1 - (bounds[3] - bounds[1])) / 2),
                  label, fill=(166, 173, 179), font=fonts["small"])
    label = f"{finger['side'][0].upper()}  {finger['finger'].upper()}"
    label_bounds = draw.textbbox((0, 0), label, font=fonts["label"])
    pad = max(4, inset * 2)
    draw.rounded_rectangle((x0 + pad, y0 + pad,
                            x0 + pad * 2 + label_bounds[2], y0 + pad * 2 + label_bounds[3]),
                           radius=4, fill=(8, 17, 20))
    draw.text((x0 + pad * 1.5, y0 + pad * 1.35), label,
              fill=(103, 237, 201), font=fonts["label"])


def _nice_limit(series):
    if not series:
        return 1.0
    peak = max(float(np.max(values)) for _, values in series)
    peak = max(peak * 1.12, 0.05)
    magnitude = 10 ** math.floor(math.log10(peak))
    return math.ceil(peak / magnitude * 2) / 2 * magnitude


def _draw_torque_tile(image, box, side, history, fonts):
    x0, y0, x1, y1 = box
    draw = ImageDraw.Draw(image)
    background = (249, 244, 238)
    ink = (174, 59, 61)
    grid = (224, 196, 187)
    draw.rectangle(box, fill=background, outline=(219, 225, 229), width=2)
    title = f"{side[0].upper()}  JOINT TORQUE"
    draw.text((x0 + 9, y0 + 6), title, fill=(151, 67, 62), font=fonts["title"])
    series = history.series(side) if history is not None else []
    left, top = x0 + 25, y0 + 38
    right, bottom = x1 - 6, y1 - 17
    if right <= left or bottom <= top:
        return
    for ratio in (0.0, 0.5, 1.0):
        y = round(top + (bottom - top) * ratio)
        draw.line((left, y, right, y), fill=grid, width=1)
    draw.line((left, top, left, bottom), fill=ink, width=2)
    draw.line((left, bottom, right, bottom), fill=ink, width=2)
    draw.polygon(((right, bottom), (right - 6, bottom - 3), (right - 6, bottom + 3)), fill=ink)
    limit = _nice_limit(series)
    draw.text((x0 + 3, top - 3), f"{limit:g}", fill=ink, font=fonts["axis"])
    draw.text((x0 + 12, bottom - 8), "0", fill=ink, font=fonts["axis"])
    draw.text((left, bottom + 3), "t-8s", fill=ink, font=fonts["axis"])
    now = history.rows[-1][0] if history is not None and history.rows else 0
    palettes = ((206, 54, 57), (231, 114, 92), (45, 127, 137),
                (113, 72, 139), (63, 73, 70))
    legend_labels = ("T", "I", "M", "R", "P")
    legend_step = max(12, round((x1 - x0) * 0.1))
    legend_x = x0 + 9
    for index, label in enumerate(legend_labels):
        draw.text((legend_x + index * legend_step, y0 + 22), label,
                  fill=palettes[index], font=fonts["axis"])
    if series:
        for finger_index in range(5):
            points = []
            for stamp, values in series:
                x = left + (right - left) * max(0.0, 1.0 - (now - stamp) / TORQUE_WINDOW_NS)
                y = bottom - float(values[finger_index]) / limit * (bottom - top)
                points.append((round(x), round(max(top, min(bottom, y)))))
            if len(points) == 1:
                draw.ellipse((points[0][0] - 1, points[0][1] - 1,
                              points[0][0] + 1, points[0][1] + 1), fill=palettes[finger_index])
            else:
                draw.line(points, fill=palettes[finger_index], width=2)
    else:
        label = "NO TORQUE DATA"
        bounds = draw.textbbox((0, 0), label, font=fonts["small"])
        draw.text(((left + right - bounds[2]) / 2, (top + bottom - bounds[3]) / 2),
                  label, fill=(151, 111, 105), font=fonts["small"])


def render_overlay(rgb, metrics, planes, torque_history=None):
    """Center-crop RGB at left and place a same-height 2-by-6 panel at right."""
    source = Image.fromarray(np.asarray(rgb, dtype=np.uint8), "RGB")
    cell_size = source.height // PANEL_ROWS
    panel_width = cell_size * PANEL_COLUMNS
    video_width = source.width - panel_width
    if video_width <= 0:
        raise ValueError("camera image is too narrow for the 2-by-6 side panel")
    crop_left = (source.width - video_width) // 2
    video = source.crop((crop_left, 0, crop_left + video_width, source.height))
    image = Image.new("RGB", source.size, (14, 17, 20))
    image.paste(video, (0, 0))
    scale = max(0.45, cell_size / 180)
    fonts = {
        "label": _font(max(7, round(12 * scale)), bold=True),
        "title": _font(max(7, round(11 * scale)), bold=True),
        "small": _font(max(7, round(9 * scale))),
        "axis": _font(max(7, round(8 * scale))),
    }
    if torque_history is None:
        torque_history = TorqueHistory()
        torque_history.update(metrics)
    for column, side in enumerate(SIDES):
        x0 = video_width + column * cell_size
        x1 = video_width + (column + 1) * cell_size - 1
        for row in range(5):
            index = column * 5 + row
            y0 = row * cell_size
            y1 = (row + 1) * cell_size - 1
            _draw_tactile_tile(image, (x0, y0, x1, y1), metrics["fingers"][index],
                               planes[index], fonts)
        _draw_torque_tile(image, (x0, 5 * cell_size, x1, source.height - 1),
                          side, torque_history, fonts)
    return image


def jpeg_bytes(image):
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()
