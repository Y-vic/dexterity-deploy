from __future__ import annotations

import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import numpy as np

from server import EpisodeData, create_server


class AnnotationServerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name) / "episode"
        (root / "timeline").mkdir(parents=True)
        np.savez_compressed(
            root / "timeline" / "clock.npz",
            elapsed_ns=np.arange(12, dtype=np.int64) * 33_333_333,
            sample_rate_hz=np.asarray(30.0),
        )
        for camera in ("ego", "left_wrist", "right_wrist"):
            camera_dir = root / "cameras" / camera
            camera_dir.mkdir(parents=True)
            (camera_dir / "rgb.mp4").write_bytes(bytes(range(256)) * 8)
            (camera_dir / "rgb_meta.json").write_text(
                json.dumps({"nominal_fps": 30, "video_codec": "h264"}),
                encoding="utf-8",
            )
            np.savez_compressed(
                camera_dir / "index.npz",
                elapsed_ns=np.arange(12, dtype=np.int64) * 33_333_333,
                frame_index=np.arange(12, dtype=np.int32),
            )
        self.episode = EpisodeData(
            root,
            Path(self.tempdir.name) / "annotations",
            make_browser_proxies=False,
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_save_ranges_builds_boolean_mask(self) -> None:
        result = self.episode.save_annotations(
            [
                {"start_frame": 2, "end_frame_exclusive": 5},
                {"start_frame": 4, "end_frame_exclusive": 7},
            ]
        )
        self.assertEqual(result["masked_frames"], 5)
        mask = np.load(self.episode.mask_npy, allow_pickle=False)
        np.testing.assert_array_equal(
            mask,
            np.asarray([True, True, False, False, False, False, False, True, True, True, True, True]),
        )

    def test_http_metadata_save_and_video_range(self) -> None:
        server = create_server("127.0.0.1", 0, self.episode)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/api/episode") as response:
                metadata = json.load(response)
            self.assertEqual(metadata["num_frames"], 12)

            body = json.dumps(
                {"ranges": [{"start_frame": 3, "end_frame_exclusive": 6}]}
            ).encode("utf-8")
            request = urllib.request.Request(
                base + "/api/annotations",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request) as response:
                saved = json.load(response)
            self.assertEqual(saved["masked_frames"], 3)

            video_request = urllib.request.Request(
                base + "/video/ego.mp4", headers={"Range": "bytes=10-19"}
            )
            with urllib.request.urlopen(video_request) as response:
                video_bytes = response.read()
                status = response.status
            self.assertEqual(status, 206)
            self.assertEqual(len(video_bytes), 10)

            head_request = urllib.request.Request(
                base + "/video/ego.mp4", method="HEAD"
            )
            with urllib.request.urlopen(head_request) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(int(response.headers["Content-Length"]), 2048)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
