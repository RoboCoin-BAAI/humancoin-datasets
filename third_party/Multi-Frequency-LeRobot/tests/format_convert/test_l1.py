"""L1 validation tests runnable without the heavy LeRobot dependencies."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from format_convert.L1.convert_l1 import (
    CAMERAS, IMU_COLUMNS, discover_sessions, inspect_session, read_timestamps, summary,
    write_sidecars,
    normalize_imu, duration_allowed,
)
from format_convert.L1.batch_convert import completed, main as batch_main, run_one


class L1ValidationTests(unittest.TestCase):
    def test_duration_boundaries_and_skip_before_existing_result(self):
        for duration, allowed in ((4.99, False), (5., True), (600., True), (600.01, False)):
            self.assertEqual(duration_allowed(duration), allowed)
        out = self.root / 'output'
        existing = out / self.session.name / 'meta'
        existing.mkdir(parents=True)
        (existing / 'info.json').write_text('{}')
        (existing / 'episodes.jsonl').write_text('{}\n')
        (existing / 'l1_source.json').write_text(json.dumps(dict(source=str(self.session))))
        with patch('format_convert.L1.batch_convert.subprocess.run', side_effect=AssertionError('must not convert')):
            self.assertEqual(run_one(self.session, out, 'ffmpeg', 'ffprobe'), 'skipped_duration')
        self.assertTrue((existing / 'info.json').exists())
        status = json.loads((out / '_batch_status' / f'{self.session.name}.json').read_text())
        self.assertEqual(status['status'], 'skipped_duration')
    def test_single_session_rejects_wrapper_before_conversion(self):
        import io
        from contextlib import redirect_stderr
        from format_convert.L1.convert_l1 import main
        for flags in ([], ['--dry-run']):
            out = self.root / 'converted'
            error = io.StringIO()
            with redirect_stderr(error), patch('format_convert.L1.convert_l1.convert') as convert:
                code = main([str(self.session.parent), '--out', str(out), *flags])
            self.assertEqual(code, 1)
            self.assertIn('Invalid session root', error.getvalue())
            self.assertIn(str(self.session.resolve()), error.getvalue())
            convert.assert_not_called()
            self.assertFalse(out.exists())

    def test_single_session_checks_required_structure_before_duration_skip(self):
        import io
        from contextlib import redirect_stderr
        from format_convert.L1.convert_l1 import main
        (self.session / 'imu/imu0.csv').unlink()
        (self.session / 'imu').rmdir()
        error = io.StringIO()
        with redirect_stderr(error):
            code = main([str(self.session), '--out', str(self.root / 'converted'), '--dry-run'])
        self.assertEqual(code, 1)
        self.assertIn('imu/ (directory)', error.getvalue())

    def test_single_session_accepts_actual_root(self):
        from format_convert.L1.convert_l1 import main
        with patch('format_convert.L1.convert_l1.session_duration', return_value=10.):
            code = main([str(self.session), '--out', str(self.root / 'converted'), '--dry-run'])
        self.assertEqual(code, 0)
        self.assertFalse((self.root / 'converted').exists())

    def test_imu_sort_and_duplicate_keep_measurements_attached(self):
        samples = [(2., [2] * 6), (1., [1] * 6), (2., [9] * 6), (3., [3] * 6)]
        cleaned, report = normalize_imu(samples)
        self.assertEqual(cleaned, [(1., [1] * 6), (2., [2] * 6), (3., [3] * 6)])
        self.assertEqual(report['duplicates_removed'], 1)
        self.assertEqual(report['backward_steps'], 1)
        self.assertEqual(report['max_backward_s'], 1.)
        self.assertEqual(samples[0], (2., [2] * 6))

    def test_imu_nonfinite_and_empty_still_rejected(self):
        for samples in ([], [(float('nan'), [0] * 6)], [(1., [float('inf')] * 6)]):
            with self.assertRaises(ValueError):
                normalize_imu(samples)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.session = self.root / "input/session/session"
        for directory in ("videos", "timestamps", "imu"):
            (self.session / directory).mkdir(parents=True)
        (self.session / "meta.json").write_text(json.dumps({
            "fps": 30, "topics": {raw: {"written_frames": 2} for raw in CAMERAS},
            "imu": {"written_samples": 3},
        }))
        for raw in CAMERAS:
            (self.session / f"videos/{raw}.mkv").write_bytes(b"placeholder")
            offset = 0.033 if raw in ("right", "bright") else 0
            (self.session / f"timestamps/{raw}_timestamps.txt").write_text(
                f"0 {100 + offset}\n1 {100 + offset + 0.033}\n")
        with (self.session / "imu/imu0.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", *IMU_COLUMNS])
            for i in range(3):
                writer.writerow([100.01 + 0.005 * i, 1, 2, 3, 4, 5, 6])

    def test_nested_discovery_and_preserved_offsets(self):
        self.assertEqual(discover_sessions(self.root / "input"), [self.session])
        records = inspect_session(self.session)
        info = summary(self.session, records)
        self.assertAlmostEqual(info["cameras"]["observation.images.head_right"]["start_s"], 0.033)
        self.assertEqual(records["imu"][0][1], [1, 2, 3, 4, 5, 6])
        self.assertEqual(CAMERAS["bleft"], "head_left_outer0")
        self.assertEqual(CAMERAS["bright"], "head_right_outer0")

    def test_timestamp_rejects_gaps_backwards_and_nonfinite(self):
        path = self.root / "times.txt"
        for content in ("0 1\n2 2\n", "0 2\n1 1\n", "0 nan\n", ""):
            with self.subTest(content=content):
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_timestamps(path)

    def test_metadata_count_mismatch_fails(self):
        (self.session / "timestamps/left_timestamps.txt").write_text("0 100\n")
        with self.assertRaisesRegex(ValueError, "metadata"):
            inspect_session(self.session)

    def test_batch_dry_run_does_not_create_output(self):
        out = self.root / "output"
        self.assertEqual(batch_main(["--source", str(self.root / "input"), "--out", str(out), "--dry-run"]), 0)
        self.assertFalse(out.exists())

    def test_resume_requires_matching_source_and_completion(self):
        out = self.root / "output"
        (out / "meta").mkdir(parents=True)
        self.assertFalse(completed(out, self.session))
        (out / "meta/info.json").write_text("{}")
        (out / "meta/episodes.jsonl").write_text("{}\n")
        marker = out / "meta/l1_source.json"
        marker.write_text(json.dumps({"source": str(self.session)}))
        self.assertTrue(completed(out, self.session))
        self.assertFalse(completed(out, self.root / "other"))

    def test_sidecars_map_camera_calibration_and_preserve_imu_topic(self):
        calibration_dir = self.session / "calibrations"
        calibration_dir.mkdir()
        camera_calibration = (
            "cam0:\n"
            "  rostopic: /fisheye/left/image_raw/compressed\n"
            "  cam_overlaps: [1]\n"
        )
        imu_calibration = (
            "imu0:\n"
            "  rostopic: /imu_data_raw\n"
            "  update_rate: 200\n"
        )
        (calibration_dir / "camera-camchain.yaml").write_text(camera_calibration)
        (calibration_dir / "camera_imu-imu.yaml").write_text(imu_calibration)
        out = self.root / "output"
        (out / "meta").mkdir(parents=True)
        records = {
            "cameras": {"left": [100.0], "right": [100.0],
                        "bleft": [100.0], "bright": [100.0]},
            "imu": [(100.0, [1, 2, 3, 4, 5, 6])],
            "t0": 100.0,
            "fps": 30.0,
            "task": "test",
        }

        write_sidecars(self.session, out, records)

        self.assertEqual(
            (out / "calibrations/camera-camchain.yaml").read_text(),
            camera_calibration,
        )
        self.assertEqual(
            (out / "calibrations/camera_imu-imu.yaml").read_text(),
            imu_calibration,
        )
        camera_mapping = json.loads(
            (out / "calibrations/lerobot_camera_mapping.json").read_text()
        )
        self.assertEqual(camera_mapping["left"]["calibration_topic"],
                         "/fisheye/left/image_raw/compressed")
        self.assertEqual(camera_mapping["left"]["feature_key"],
                         "observation.images.head_left")
        self.assertEqual(camera_mapping["bleft"]["feature_key"],
                         "observation.images.head_left_outer0")

    def test_sidecars_preserve_task_annotations_and_encoder_log(self):
        (self.session / "calibrations").mkdir()
        annotation_files = {
            "任务详情.txt": "任务详情原文\n步骤一\n",
            "厨房_config.yaml": "task: 测试任务\nscene:\n  notes: 原始标注\n",
            "encoder.log": "encoder output\n",
        }
        expected = {}
        for name, content in annotation_files.items():
            path = self.session / name
            path.write_text(content, encoding="utf-8")
            expected[name] = path.read_bytes()

        out = self.root / "output"
        (out / "meta").mkdir(parents=True)
        write_sidecars(self.session, out, {
            "cameras": {"left": [100.0], "right": [100.0],
                        "bleft": [100.0], "bright": [100.0]},
            "imu": [(100.0, [1, 2, 3, 4, 5, 6])],
            "t0": 100.0,
            "fps": 30.0,
            "task": "测试任务",
        })

        for name, content in expected.items():
            with self.subTest(name=name):
                self.assertEqual(
                    (out / "annotation" / name).read_bytes(),
                    content,
                )


if __name__ == "__main__":
    unittest.main()
