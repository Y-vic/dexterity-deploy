"""Optional Linux usbmon metadata-only collector. Never copies USB payloads.

Run separately with permission to read /dev/usbmon0; the robot nodes need no
extra privileges. ABI: https://docs.kernel.org/usb/usbmon.html
"""
import argparse
from collections import defaultdict
import ctypes
import fcntl
import json
import os
from pathlib import Path
import select
import struct
import tempfile
import time


def aggregate_header(header, counters):
    if len(header) < 48:
        raise ValueError("short usbmon header")
    event, transfer, endpoint, device = struct.unpack_from("=BBBB", header, 8)
    bus = struct.unpack_from("=H", header, 12)[0]
    status = struct.unpack_from("=i", header, 28)[0]
    length = struct.unpack_from("=I", header, 32)[0]
    key = f"{bus}:{device}:{endpoint & 127}:{'in' if endpoint & 128 else 'out'}:{transfer}"
    row = counters.setdefault(key, {"completed": 0, "actual_bytes": 0, "errors": 0, "iso_packet_errors": 0})
    if event == ord("C"):
        row["completed"] += 1
        row["actual_bytes"] += length
        row["errors"] += int(status != 0)
        if transfer == 0:
            row["iso_packet_errors"] += max(0, struct.unpack_from("=i", header, 40)[0])
    elif event == ord("E"):
        row["errors"] += 1


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".usbmon-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle)
        os.chmod(temporary, 0o644)  # Aggregates only; readable by the recorder.
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def collect(device, output):
    # ioctl argument holds addresses of the 64-byte header and a NULL payload
    # buffer with alloc=0: no image, keyboard or network payload enters Python.
    header = ctypes.create_string_buffer(64)
    argument = struct.pack("@PPN", ctypes.addressof(header), 0, 0)
    getx = (1 << 30) | (len(argument) << 16) | (0x92 << 8) | 10
    stats_request = (2 << 30) | (8 << 16) | (0x92 << 8) | 3
    fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    counters = {}
    previous = {}
    dropped_total = 0
    started = time.monotonic()
    previous_at = started
    try:
        while True:
            if select.select([fd], [], [], 0.1)[0]:
                # Bound each batch so the once-per-second report is not starved.
                for _ in range(2048):
                    try:
                        fcntl.ioctl(fd, getx, argument)
                    except BlockingIOError:
                        break
                    aggregate_header(header.raw, counters)
            now = time.monotonic()
            if now - previous_at < 1:
                continue
            statbuf = bytearray(8)
            fcntl.ioctl(fd, stats_request, statbuf, True)
            queued, dropped = struct.unpack("=II", statbuf)
            dropped_total += dropped
            buses = defaultdict(float)
            for key, row in counters.items():
                delta = row["actual_bytes"] - previous.get(key, 0)
                buses[key.split(":")[0]] += delta * 8 / (now-previous_at) / 1e6
                previous[key] = row["actual_bytes"]
            atomic_json(output, {
                "timestamp_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
                "monitor_started_monotonic_ns": int(started*1e9),
                "window_s": now-previous_at,
                "scope": "completed_URB_payload_bytes_not_wire_utilization",
                "payload_captured": False, "bus_payload_mbps": dict(buses),
                "monitor_dropped_total": dropped_total, "monitor_queued": queued,
                "endpoints": counters,
            })
            previous_at = now
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/usbmon0")
    parser.add_argument("--output", default="/tmp/ur_sharpa_usbmon.json")
    args = parser.parse_args()
    try:
        collect(args.device, args.output)
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        parser.exit(1, f"usbmon unavailable: {exc}\nLoad usbmon and run this collector with device read permission.\n")


if __name__ == "__main__":
    main()
