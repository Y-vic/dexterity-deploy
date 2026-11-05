"""Append received ROS messages independently of the 30 Hz sample filter.

The database stores CDR bytes, including invalid values, without changing the
message contract. Sequence numbers count recorder receipts, NOT device frames.
Memory is bounded; overload is a recorded failure, never a silent overwrite.
"""

from collections import Counter
import json
from pathlib import Path
import queue
import sqlite3
import threading
import time


class MessageJournal:
    def __init__(self, directory, max_pending_bytes=64 * 1024 * 1024):
        if max_pending_bytes <= 0:
            raise ValueError("journal memory budget must be positive")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_pending_bytes = int(max_pending_bytes)
        self._queue = queue.Queue()
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._accepting = True
        self._pending_bytes = 0
        self._peak_pending_bytes = 0
        self._received = Counter()
        self._written = Counter()
        self._rejected = Counter()
        self._errors = []
        self._worker_failed = False
        self._commits = self._max_batch_size = 0
        self._commit_ms_max = 0.
        self._thread = threading.Thread(target=self._run, name="raw-message-journal", daemon=True)
        self._thread.start()
        if not self._ready.wait(5):
            self.close(timeout=0)
            raise RuntimeError("raw message journal initialization timed out")
        if self._errors:
            self.close()
            raise RuntimeError(self._errors[0])

    def _error(self, message):
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)

    def fail(self, message):
        """Latch capture/serialization failures in the final completeness result."""
        self._error(str(message))

    def append(self, topic, type_name, payload, source_timestamp_ns=0,
               received_unix_ns=None, received_monotonic_ns=None, inference_json=None):
        received_unix_ns = time.time_ns() if received_unix_ns is None else int(received_unix_ns)
        received_monotonic_ns = (time.monotonic_ns() if received_monotonic_ns is None
                                 else int(received_monotonic_ns))
        payload = bytes(payload)
        # Account for metadata/object overhead as well as payload bytes.
        cost = (len(payload) + len(topic.encode()) + len(type_name.encode()) + 512
                + (len(inference_json.encode('utf-8')) if inference_json is not None else 0))
        with self._lock:
            if not self._accepting:
                return False
            self._received[topic] += 1
            sequence = self._received[topic]
            if self._worker_failed or self._pending_bytes + cost > self.max_pending_bytes:
                self._rejected[topic] += 1
                reason = "raw message journal memory budget exceeded"
                if not self._worker_failed and reason not in self._errors:
                    self._errors.append(reason)
                return False
            self._pending_bytes += cost
            self._peak_pending_bytes = max(self._peak_pending_bytes, self._pending_bytes)
            self._queue.put_nowait((cost, (topic, type_name, sequence,
                received_unix_ns, received_monotonic_ns, int(source_timestamp_ns), payload,
                inference_json)))
            return True

    @staticmethod
    def _write_batch(connection, batch):
        with connection:
            connection.executemany("INSERT INTO messages VALUES (NULL,?,?,?,?,?,?,?)",
                                   [row[:7] for _, row in batch])

    def _run(self):
        connection = None
        inference_file = None
        try:
            connection = sqlite3.connect(self.directory / "messages.sqlite3")
            # Keep FULL durability while avoiding rollback-journal creation and
            # deletion/fsync on each 5 ms batch. close() checkpoints the WAL.
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("""CREATE TABLE messages (
                id INTEGER PRIMARY KEY, topic TEXT NOT NULL, type_name TEXT NOT NULL,
                receive_sequence INTEGER NOT NULL, received_unix_ns INTEGER NOT NULL,
                received_monotonic_ns INTEGER NOT NULL, source_timestamp_ns INTEGER NOT NULL,
                cdr BLOB NOT NULL)""")
            connection.commit()
            self._ready.set()
            stopping = False
            while not stopping:
                entry = self._queue.get()
                if entry is None:
                    break
                batch = [entry]
                # At ~900 execution events/s, flushing every isolated arrival
                # turns FULL durability into hundreds of tiny fsyncs/s. Gather
                # up to 5 ms in this background thread; callbacks never wait.
                deadline = time.monotonic()+.005
                while len(batch) < 128:
                    try:
                        remaining = deadline-time.monotonic()
                        entry = (self._queue.get(timeout=remaining) if remaining > 0
                                 else self._queue.get_nowait())
                    except queue.Empty:
                        break
                    if entry is None:
                        stopping = True
                        break
                    batch.append(entry)
                began = time.perf_counter()
                self._write_batch(connection, batch)
                commit_ms = (time.perf_counter()-began)*1000.
                for _, row in batch:
                    if row[7] is not None:
                        if inference_file is None:
                            inference_file = (self.directory / 'policy_inference.jsonl').open('w', encoding='utf-8')
                        inference_file.write(row[7] + '\n')
                if inference_file is not None:
                    inference_file.flush()
                with self._lock:
                    self._commits += 1
                    self._max_batch_size = max(self._max_batch_size, len(batch))
                    self._commit_ms_max = max(self._commit_ms_max, commit_ms)
                    for cost, row in batch:
                        self._written[row[0]] += 1
                        self._pending_bytes -= cost
        except Exception as exc:
            with self._lock:
                self._worker_failed = True
            self._error(f"raw message journal write failed: {exc}")
        finally:
            if inference_file is not None:
                try:
                    inference_file.close()
                except Exception as exc:
                    self._error(f'policy inference file close failed: {exc}')
            if connection is not None:
                try:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    # Archive a standalone database. A later read-only audit
                    # must not create -wal/-shm sidecars inside the episode.
                    mode = connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
                    if mode != "delete":
                        raise RuntimeError("could not finalize standalone message database")
                except Exception as exc:
                    self._error(f"raw message journal finalize failed: {exc}")
                finally:
                    connection.close()
            self._ready.set()

    def status(self):
        with self._lock:
            received, written = dict(self._received), dict(self._written)
            return {
                "schema": "ur_sharpa_received_messages.v1",
                "encoding": "cdr", "path": "messages.sqlite3",
                "policy_inference_path": "policy_inference.jsonl" if received.get('/policy/inference') else None,
                "sequence_scope": "recorder_receipt_per_topic",
                "received": received, "written": written,
                "rejected": dict(self._rejected),
                "pending_bytes": self._pending_bytes,
                "peak_pending_bytes": self._peak_pending_bytes,
                "max_pending_bytes": self.max_pending_bytes,
                "commits": self._commits, "max_batch_size": self._max_batch_size,
                "commit_ms_max": round(self._commit_ms_max, 3),
                "errors": list(self._errors),
                "closed": not self._accepting and not self._thread.is_alive(),
                "complete": (not self._accepting and not self._thread.is_alive()
                             and not self._errors and received == written),
            }

    def close(self, timeout=30.0):
        with self._lock:
            if self._accepting:
                self._accepting = False
                self._queue.put_nowait(None)
        self._thread.join(timeout=max(0.0, timeout))
        if self._thread.is_alive():
            # Caller must leave the episode in place and retry; never rename
            # or delete a database while its owning thread is still writing.
            return False
        (self.directory / "messages_meta.json").write_text(
            json.dumps(self.status(), indent=2), encoding="utf-8")
        return True
