import tempfile
import unittest
from pathlib import Path
import csv
import gzip
import json
import struct

import numpy as np

from wifi_datahub.standardize import resample_csi
from wifi_datahub.adapters.wallhack import parse_interleaved_imag_real, select_subcarriers
from wifi_datahub.adapters.aril import normalize_aril_arrays
from wifi_datahub.adapters.ut_har import normalize_ut_har_arrays
from wifi_datahub.quality import inspect_npz
from wifi_datahub.adapters.xrf_v2 import convert_xrf_v2_h5, normalize_xrf_v2_wifi
from wifi_datahub.prepare import prepare_dataset
from wifi_datahub.prepare import registered_datasets
from wifi_datahub.catalog import load_datasets
from wifi_datahub.splits import generate_split, infer_metadata
from wifi_datahub.views import ViewOptions, create_standard_view, resolve_task_profile
from wifi_datahub.adapters.official_profiles import (
    convert_csi_bench_mat, convert_mmfi_directory, convert_ntu_fi_mat,
    convert_signfi_mat, convert_three_rooms_directory, convert_widar_csv,
    convert_wimans, convert_xrf55_npy, convert_ehunam_mat,
    convert_wifi_presence_json,
    convert_wifi_tad_npy,
    convert_operanet_mat,
    convert_nist_breathesmart,
    convert_csida_zarr,
    convert_exposing_csi_mat,
    convert_wifi_80mhz_mat,
    convert_usrp_amplitude_csv,
    convert_wireless_har_wifi,
)
from wifi_datahub.adapters.intel5300 import convert_wiar_dat, read_bf_file
from wifi_datahub.adapters.canonical import canonicalize_csi_output


class StandardizeTests(unittest.TestCase):
    def test_vital_sign_profile_is_thirty_seconds(self):
        profile = resolve_task_profile("vital-sign")
        self.assertEqual(profile.target_rate_hz, 10.0)
        self.assertEqual(profile.duration_s, 30.0)
        self.assertEqual(profile.target_rate_hz * profile.duration_s, 300.0)

    def test_widar_bvp_profile_has_no_fictitious_rate(self):
        profile = resolve_task_profile("semantic-bvp")
        self.assertIsNone(profile.target_rate_hz)
        self.assertIsNone(profile.duration_s)
        self.assertEqual(profile.target_length, 22)
        self.assertEqual(profile.interpolation, "none")

    def test_fixed_rate_and_duration(self):
        timestamp = np.asarray([0.0, 0.02, 0.04, 0.06, 0.08])
        real = np.ones((5, 1, 2), dtype=np.float32)
        imag = np.zeros_like(real)
        result = resample_csi(timestamp, real, imag, sample_rate_hz=100, duration_s=0.1)
        self.assertEqual(result["csi_real"].shape, (10, 1, 2))
        self.assertEqual(result["timestamp_s"][-1], 0.09)
        self.assertTrue(np.allclose(result["power_db_rel"][result["valid_mask"]], 0.0))
        self.assertFalse(result["valid_mask"][-1])

    def test_sorts_and_deduplicates_timestamps(self):
        timestamp = np.asarray([0.02, 0.0, 0.02, 0.04])
        real = np.arange(4, dtype=np.float32).reshape(4, 1, 1)
        imag = np.zeros_like(real)
        result = resample_csi(timestamp, real, imag, sample_rate_hz=50, duration_s=0.06)
        self.assertEqual(result["csi_real"].shape[0], 3)
        self.assertTrue(np.all(np.diff(result["timestamp_s"]) > 0))

    def test_wallhack_iq_order_and_subcarriers(self):
        parsed = parse_interleaved_imag_real("[1, 2, 3, 4]")
        self.assertTrue(np.allclose(parsed, [2 + 1j, 4 + 3j]))
        selected, indices = select_subcarriers(np.arange(128))
        self.assertEqual(selected.size, 52)
        self.assertEqual(indices[0], 6)

    def test_aril_official_shape(self):
        data = np.zeros((3, 52, 192), dtype=np.float64)
        amplitude, activity, location = normalize_aril_arrays(data, [[0], [1], [2]], [[3], [4], [5]])
        self.assertEqual(amplitude.shape, (3, 192, 52, 1, 1))
        self.assertEqual(amplitude.dtype, np.float32)
        self.assertEqual(activity.tolist(), [0, 1, 2])
        self.assertEqual(location.tolist(), [3, 4, 5])

    def test_prepare_aril_uses_explicit_subcarrier_tx_rx_axes(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aril" / "original" / "train_data_split_amp.mat"
            source.parent.mkdir(parents=True)
            savemat(source, {
                "train_data": np.zeros((2, 52, 192), dtype=np.float32),
                "train_activity_label": np.asarray([[0], [1]], dtype=np.int16),
                "train_location_label": np.asarray([[2], [3]], dtype=np.int16),
            })
            summary = prepare_dataset("aril", root, setting="random")
            self.assertEqual(summary["view_options"]["target_rate_hz"], 100.0)
            self.assertEqual(summary["window_policy"]["selected_window_length"], 192)
            output = root / "aril" / "standardized" / "views" / "train_data_split_amp.npz"
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (2, 192, 52, 1, 1))
                self.assertEqual(loaded["subcarrier_mask"].shape, (52,))
                self.assertEqual(loaded["tx_link_mask"].tolist(), [True])
                self.assertEqual(loaded["rx_link_mask"].tolist(), [True])
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["axis_order"], ["sample", "time", "subcarrier", "tx_link", "rx_link"])
            self.assertTrue(metadata["target_rate_assumed"])

    def test_ut_har_official_shape(self):
        data = np.zeros((2, 250, 90), dtype=np.float64)
        amplitude, labels = normalize_ut_har_arrays(data, [1, 6])
        self.assertEqual(amplitude.shape, (2, 250, 30, 1, 3))
        self.assertEqual(labels.tolist(), [1, 6])

    def test_xrf_v2_official_wifi_shape(self):
        data = np.zeros((2900, 3, 3, 30), dtype=np.float64)
        amplitude, receivers = normalize_xrf_v2_wifi(data, [0, 2])
        self.assertEqual(amplitude.shape, (2900, 30, 3, 2))
        self.assertEqual(amplitude.dtype, np.float32)
        self.assertEqual(receivers, [0, 2])

    def test_xrf_v2_sidecar_keeps_original_and_canonical_axes(self):
        import h5py
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "3_2_6.h5"
            output = Path(directory) / "3_2_6.npz"
            with h5py.File(source, "w") as handle:
                handle.create_dataset("amp", data=np.zeros((8, 3, 3, 30), dtype=np.float32))
                handle.create_dataset("pha", data=np.zeros((8, 3, 3, 30), dtype=np.float32))
                handle.create_dataset("label", data=np.asarray([[1, 2, 0, 8]], dtype=np.int16))
            convert_xrf_v2_h5(source, output)
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_shape"], [8, 3, 3, 30])
            self.assertEqual(metadata["source_axis_order"], ["packet", "rx_link", "tx_link", "subcarrier"])
            self.assertEqual(metadata["axis_order"], ["packet", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(metadata["sample_rate_hz"], 50.0)
            self.assertEqual(metadata["segments"][0]["label"], "writing")
            self.assertEqual(metadata["segments"][0]["end_seconds"], 0.16)
            self.assertEqual(metadata["settings"]["subject_id"], 3)
            self.assertEqual(metadata["settings"]["scene"], "study room")
            self.assertEqual(metadata["raw_activity_count"], 34)
            self.assertEqual(metadata["derived_benchmark_activity_count"], 30)

    def test_prepare_wallhack_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wallhack18k" / "original" / "LOS" / "BQ" / "w1.csv"
            source.parent.mkdir(parents=True)
            vector = [value for index in range(128) for value in (index * 0.1, index * 0.2)]
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["data", "class"])
                writer.writeheader()
                writer.writerow({"data": str(vector), "class": "1"})
                writer.writerow({"data": str(vector), "class": "1"})
            summary = prepare_dataset("wallhack18k", root, setting="random")
            self.assertEqual(summary["converted"], 1)
            output = root / "wallhack18k" / "standardized" / "LOS__BQ__w1.npz"
            report = root / "wallhack18k" / "reports" / "LOS__BQ__w1.quality.json"
            self.assertTrue(output.exists())
            self.assertTrue(report.exists())

    def test_prepare_ut_har_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ut-har" / "original" / "processed.npz"
            source.parent.mkdir(parents=True)
            np.savez(source, data=np.zeros((2, 250, 90)), label=np.asarray([0, 1]))
            summary = prepare_dataset("ut-har", root, setting="random")
            self.assertEqual(summary["converted"], 1)
            output = root / "ut-har" / "standardized" / "processed.npz"
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 250, 30, 1, 3))
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["axis_order"],
                ["sample", "time", "subcarrier", "tx_link", "rx_link"],
            )

    def test_standard_view_target_length_and_flat_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "native.npz"
            np.savez(
                source,
                amplitude=np.arange(2 * 4 * 3 * 2, dtype=np.float32).reshape(2, 4, 3, 2),
                valid_mask=np.ones((2, 4), dtype=bool),
                packet_index=np.arange(4, dtype=np.int32),
            )
            source.with_suffix(".json").write_text(json.dumps({
                "dataset_id": "synthetic", "standard_representation": "amplitude",
                "shape": [2, 4, 3, 2], "axis_order": ["sample", "time", "link", "subcarrier"],
                "sample_rate_hz": 4.0, "transformations": [],
            }), encoding="utf-8")
            output = root / "view.npz"
            create_standard_view(source, output, ViewOptions(target_length=8, layout="link-subcarrier", links=3, subcarriers=2))
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 8, 6))
            self.assertEqual(loaded["valid_mask"].shape, (2, 8))
            meta = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(meta["axis_order"], ["sample", "time", "feature"])

    def test_prepare_can_emit_derived_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ut-har" / "original" / "processed.npz"
            source.parent.mkdir(parents=True)
            np.savez(source, data=np.zeros((2, 250, 90)), label=np.asarray([0, 1]))
            summary = prepare_dataset(
                "ut-har", root, setting="random",
                view_options=ViewOptions(target_length=128, layout="link-subcarrier"),
            )
            self.assertEqual(summary["converted"], 1)
            output = root / "ut-har" / "standardized" / "views" / "processed.npz"
            self.assertTrue(output.exists())
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (4, 128, 90))
            self.assertEqual(int(loaded["valid_mask"].sum()), 500)
            self.assertEqual(loaded["source_sample_index"].tolist(), [0, 0, 1, 1])
            self.assertEqual(loaded["window_valid_length"].tolist(), [128, 122, 128, 122])
            self.assertTrue(summary["records"][0]["native_output"].endswith("standardized/processed.npz"))
            self.assertTrue(summary["window_policy"]["full_source_preserved"])

    def test_prepare_uses_shortest_dataset_clip_as_default_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "wallhack18k" / "original" / "LOS" / "BQ"
            original.mkdir(parents=True)
            vector = [value for index in range(128) for value in (index * 0.1, index * 0.2)]
            for name, length in (("short.csv", 2), ("long.csv", 5)):
                with (original / name).open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=["data", "class"])
                    writer.writeheader()
                    for _ in range(length):
                        writer.writerow({"data": str(vector), "class": "1"})
            summary = prepare_dataset("wallhack18k", root, setting="random")
            self.assertEqual(summary["window_policy"]["selected_window_length"], 2)
            long_view = np.load(root / "wallhack18k" / "standardized" / "views" / "LOS__BQ__long.npz")
            self.assertEqual(long_view["amplitude"].shape[0:2], (3, 2))
            self.assertEqual(int(long_view["valid_mask"].sum()), 5)
            self.assertEqual(long_view["window_valid_length"].tolist(), [2, 2, 1])

    def test_ut_har_official_sensefi_x_y_filename_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "ut-har" / "original" / "UT_HAR" / "data" / "X_test.csv"
            label_path = root / "ut-har" / "original" / "UT_HAR" / "label" / "y_test.csv"
            data_path.parent.mkdir(parents=True)
            label_path.parent.mkdir(parents=True)
            with data_path.open("wb") as handle:
                np.save(handle, np.zeros((2, 250, 90), dtype=np.float32))
            with label_path.open("wb") as handle:
                np.save(handle, np.asarray([0, 1], dtype=np.int64))
            summary = prepare_dataset("ut-har", root, setting="official")
            self.assertEqual(summary["converted"], 1)
            output = root / "ut-har" / "standardized" / "UT_HAR__data__X_test.npz"
            self.assertEqual(np.load(output)["amplitude"].shape, (2, 250, 30, 1, 3))
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["source_shape"], [2, 250, 90])
            self.assertEqual(metadata["antenna_layout"], {"tx_links": 1, "rx_links": 3})
            self.assertEqual(summary["split"]["partition_counts"]["test"], 2)

    def test_all_catalog_datasets_have_prepare_adapters(self):
        self.assertEqual(registered_datasets(), sorted(item["id"] for item in load_datasets()))

    def test_wireless_har_adapter_and_random_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wireless-har-wifi-uwb" / "original" / "Wireless_sensing_human_activity_recognition" / "WiFi_CSI" / "Room_1" / "processed.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.zeros((10, 20, 30), dtype=np.float32))
            summary = prepare_dataset("wireless-har-wifi-uwb", root, setting="random", seed=7)
            self.assertEqual(summary["split"]["partition_counts"], {"train": 1, "val": 0, "test": 0})
            output = root / "wireless-har-wifi-uwb" / "standardized" / "Wireless_sensing_human_activity_recognition__WiFi_CSI__Room_1__processed.npz"
            # The synthetic source exposes 20 flattened streams but no reliable
            # Tx/Rx factorization, so every stream is preserved as 1 Tx × 20 Rx.
            self.assertEqual(np.load(output)["amplitude"].shape, (10, 30, 1, 20))

    def test_group_holdout_has_no_subject_leakage(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory) / "wiar"
            records = []
            for subject in (1, 2, 3):
                source = dataset_root / "original" / f"subject_{subject}" / "clip.npy"
                output = dataset_root / "standardized" / f"subject_{subject}.npz"
                source.parent.mkdir(parents=True, exist_ok=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(source, np.zeros((2, 3)))
                np.savez(output, amplitude=np.zeros((2, 1, 3)))
                output.with_suffix(".json").write_text(
                    '{"shape":[2,1,3],"axis_order":["time","link","subcarrier"]}', encoding="utf-8"
                )
                records.append({"source": str(source), "output": str(output), "status": "converted"})
            result = generate_split("wiar", dataset_root, records, "cross_subject", holdout=["3"])
            self.assertEqual(result["groups"]["test"], ["3"])
            self.assertNotIn("3", result["groups"]["train"] + result["groups"]["val"])

    def test_mmfi_cross_subject_reads_official_directory_ids(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for subject in (1, 2, 3):
                source = root / "mm-fi" / "original" / "E01" / f"S{subject:02d}" / "A01" / "wifi-csi"
                source.mkdir(parents=True)
                # One authentic MM-Fi frame is [3 Rx, 114 subcarriers,
                # 10 packet captures]. Keep the fixture faithful so this test
                # exercises subject splitting rather than invalid input.
                savemat(source / "frame001.mat", {"CSIamp": np.zeros((3, 114, 10), dtype=np.float32)})
            summary = prepare_dataset("mm-fi", root, setting="cross_subject", holdout=["3"])
            split_path = Path(summary["split"]["manifest"])
            split = json.loads(split_path.read_text(encoding="utf-8"))
            self.assertEqual(split["groups"]["test"], ["3"])
            self.assertEqual(split["partition_counts"]["test"], 1)

    def test_predefined_split_uses_release_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset_root = Path(directory) / "ntu-fi"
            records = []
            for split in ("train", "test"):
                source = dataset_root / "original" / split / "samples.npy"
                output = dataset_root / "standardized" / f"{split}.npz"
                source.parent.mkdir(parents=True, exist_ok=True)
                output.parent.mkdir(parents=True, exist_ok=True)
                np.save(source, np.zeros((2, 3)))
                np.savez(output, amplitude=np.zeros((2, 1, 3)))
                output.with_suffix(".json").write_text(
                    '{"shape":[2,1,3],"axis_order":["time","link","subcarrier"]}', encoding="utf-8"
                )
                records.append({"source": str(source), "output": str(output), "status": "converted"})
            result = generate_split("ntu-fi", dataset_root, records, "official")
            self.assertEqual(result["partition_counts"], {"train": 1, "val": 0, "test": 1})

    def test_csi_bench_official_mat_profile(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "human.mat", Path(directory) / "human.npz"
            savemat(source, {"X": np.zeros((2, 250, 100), dtype=np.float32)})
            convert_csi_bench_mat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 250, 100, 1, 1))
            self.assertEqual(loaded["subcarrier_index"].shape, (100,))
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["axis_order"],
                ["sample", "time", "subcarrier", "tx_link", "rx_link"],
            )
            self.assertEqual(metadata["antenna_mapping"], "single_tx_flattened_rx")

    def test_csi_bench_task_auto_profiles_and_native_dimensions(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "csi-bench" / "original"
            breathing = original / "BreathingDetection" / "sub_Human" / "act_sleep" / "breath.mat"
            fall = original / "FallDetection" / "sub_Human" / "act_Fall" / "fall.mat"
            breathing.parent.mkdir(parents=True)
            fall.parent.mkdir(parents=True)
            savemat(breathing, {"CSI_amps": np.zeros((64, 240, 1), dtype=np.float32)})
            savemat(fall, {"CSI_amps": np.zeros((64, 500, 1), dtype=np.float32)})
            summary = prepare_dataset("csi-bench", root, setting="random")
            self.assertEqual(summary["view_options"]["profile"], "task-auto")
            self.assertEqual(summary["task_profiles"]["BreathingDetection"]["target_rate_hz"], 10.0)
            self.assertEqual(summary["task_profiles"]["BreathingDetection"]["selected_window_length"], 240)
            self.assertEqual(summary["task_profiles"]["FallDetection"]["target_rate_hz"], 100.0)
            self.assertEqual(summary["task_profiles"]["FallDetection"]["selected_window_length"], 300)
            breath_view = np.load(root / "csi-bench" / "standardized" / "views" / "BreathingDetection__sub_Human__act_sleep__breath.npz")
            fall_view = np.load(root / "csi-bench" / "standardized" / "views" / "FallDetection__sub_Human__act_Fall__fall.npz")
            self.assertEqual(breath_view["amplitude"].shape, (240, 64, 1, 1))
            self.assertEqual(fall_view["amplitude"].shape, (2, 300, 64, 1, 1))
            self.assertTrue(np.all(breath_view["subcarrier_mask"]))

    def test_csi_bench_dimension_request_adds_masks(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "BreathingDetection" / "act_sleep" / "sample.mat"
            source.parent.mkdir(parents=True)
            native = root / "native.npz"
            output = root / "view.npz"
            savemat(source, {"CSI_amps": np.zeros((64, 80, 1), dtype=np.float32)})
            convert_csi_bench_mat(source, native)
            create_standard_view(
                native, output,
                ViewOptions(target_length=80, subcarriers=20, tx_links=2, rx_links=3),
            )
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (80, 20, 2, 3))
            self.assertEqual(loaded["subcarrier_mask"].tolist(), [True] * 20)
            self.assertEqual(loaded["tx_link_mask"].tolist(), [True, False])
            self.assertEqual(loaded["rx_link_mask"].tolist(), [True, False, False])

    def test_mmfi_official_frame_directory(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "E01" / "S01" / "A01" / "wifi-csi"
            source.mkdir(parents=True)
            first = np.zeros((3, 114, 10))
            first[0, 0, 0] = np.inf
            savemat(source / "frame001.mat", {"CSIamp": first})
            savemat(source / "frame002.mat", {"CSIamp": np.ones((3, 114, 10))})
            output = Path(directory) / "out.npz"
            convert_mmfi_directory(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (20, 114, 1, 3))
            self.assertTrue(np.isfinite(loaded["amplitude"]).all())
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["antenna_layout"], {"tx_links": 1, "rx_links": 3})

    def test_ntu_fi_official_mat_profile(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "walk.mat", Path(directory) / "walk.npz"
            values = np.arange(3 * 114 * 20, dtype=np.float32).reshape(3, 114, 20)
            savemat(source, {"CSIamp": values})
            convert_ntu_fi_mat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (5, 114, 1, 3))
            np.testing.assert_array_equal(loaded["amplitude"][:, :, 0, 2], values[2, :, ::4].T)
            np.testing.assert_allclose(
                loaded["official_normalized_amplitude"],
                (loaded["amplitude"] - 42.3199) / 4.9802,
                rtol=1e-6,
            )
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source_shape"], [3, 114, 20])
            self.assertEqual(metadata["axis_order"], ["time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(metadata["antenna_layout"], {"tx_links": 1, "rx_links": 3})

    def test_widar_official_bvp_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "1-Push&Pull" / "user2-1-5-4-8-1.csv"
            source.parent.mkdir()
            output = Path(directory) / "gesture.npz"
            np.savetxt(source, np.arange(8800).reshape(22, 400), delimiter=",")
            convert_widar_csv(source, output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["bvp"].shape, (22, 20, 20))
                np.testing.assert_allclose(
                    loaded["official_normalized_bvp"],
                    (loaded["bvp"] - 0.0025) / 0.0119,
                )
                self.assertEqual(str(loaded["activity_label"]), "Push&Pull")
                self.assertEqual(str(loaded["subject"]), "user2")
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["time_axis"], "time_bin")
            self.assertEqual(metadata["source_shape"], [22, 400])
            self.assertEqual(
                metadata["axis_order"],
                ["time_bin", "velocity_x_bin", "velocity_y_bin"],
            )
            self.assertIn("canonical_tensor_exception", metadata)
            view = Path(directory) / "view.npz"
            create_standard_view(output, view, ViewOptions(
                target_rate_hz=100.0, target_length=20, profile="general-sensing",
            ))
            view_metadata = json.loads(view.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertIsNone(view_metadata["sample_rate_hz"])
            self.assertTrue(view_metadata["target_rate_ignored_for_semantic_bins"])
            self.assertEqual(
                view_metadata["axis_order"][-3:],
                ["time_bin", "velocity_x_bin", "velocity_y_bin"],
            )

    def test_three_rooms_official_csv_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "data.csv"
            np.savetxt(source, np.zeros((3, 114 * 9)), delimiter=",")
            np.savetxt(source.with_name("label.csv"), np.asarray([[0, 1], [1, 2], [2, 3]]), delimiter=",")
            output = Path(directory) / "out.npz"
            convert_three_rooms_directory(source, output)
            self.assertEqual(np.load(output)["amplitude"].shape, (3, 4, 114))

    def test_signfi_official_mat_axes(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "dataset_lab_276_dl.mat", Path(directory) / "out.npz"
            savemat(source, {"csid_lab": np.zeros((200, 30, 3, 2), dtype=np.complex64), "label_lab": [[1], [2]]})
            convert_signfi_mat(source, output)
            archive = np.load(output)
            self.assertEqual(archive["amplitude"].shape, (2, 200, 30, 1, 3))
            self.assertEqual(archive["phase_rad"].shape, (2, 200, 30, 1, 3))
            metadata = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(metadata["axis_order"], ["sample", "time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(metadata["source_shape"], [200, 30, 3, 2])
            self.assertIsNone(metadata["sample_rate_hz"])

    def test_wimans_and_xrf55_official_npy_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "act_1_1.npy"
            np.save(source, np.zeros((10, 3, 3, 30), dtype=np.float32))
            wimans_output, xrf_output = Path(directory) / "wimans.npz", Path(directory) / "xrf.npz"
            convert_wimans(source, wimans_output)
            self.assertEqual(np.load(wimans_output)["amplitude"].shape, (10, 30, 3, 3))
            wimans_sidecar = json.loads(wimans_output.with_suffix(".json").read_text())
            self.assertEqual(wimans_sidecar["axis_order"], ["time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(wimans_sidecar["source_axis_order"], ["time", "tx_link", "rx_link", "subcarrier"])
            self.assertAlmostEqual(wimans_sidecar["sample_rate_hz"], 10 / 3)
            self.assertEqual(wimans_sidecar["nominal_sample_rate_hz"], 1000.0)
            self.assertEqual(wimans_sidecar["fixed_clip_duration_s"], 3.0)
            xrf_source = Path(directory) / "01_01_01.npy"
            np.save(xrf_source, np.zeros((270, 1000), dtype=np.float64))
            convert_xrf55_npy(xrf_source, xrf_output)
            self.assertEqual(np.load(xrf_output)["amplitude"].shape, (1000, 30, 1, 9))
            sidecar = json.loads(xrf_output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["sample_rate_hz"], 200.0)
            self.assertEqual(sidecar["label_sets"]["Activity"], ["carrying weight"])
            self.assertEqual(sidecar["source_shape"], [270, 1000])
            self.assertEqual(sidecar["antenna_layout"], {"tx_links": 1, "rx_links": 9})
            self.assertEqual(
                sidecar["receiver_grouping"],
                {"receiver_devices": 3, "receiving_antennas_per_device": 3},
            )

    def test_ehunam_official_subcarrier_removal(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "MC1_01A_1_HAR_e_J_#_#_01.mat", Path(directory) / "out.npz"
            savemat(source, {
                "CSI": np.ones((4, 64), dtype=np.complex64), "BW": [[20]],
                "Subcarriers": [[64]], "Environment": "Office", "TimeStamp": [[0], [.1], [.2], [.3]],
            })
            convert_ehunam_mat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (4, 1, 56))
            self.assertTrue(np.allclose(loaded["timestamp_s"], [0, .1, .2, .3]))
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(sidecar["sample_rate_hz"], 10.0)
            self.assertEqual(sidecar["sample_rate_provenance"], "inferred from the released TimeStamp span")

    def test_presence_movement_official_json_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "segments" / "mobile" / "G19-10.csi.json.gz"
            source.parent.mkdir(parents=True)
            output = Path(directory) / "out.npz"
            packet = [[{"r": subcarrier + link, "i": -link} for link in range(3)] for subcarrier in range(30)]
            with gzip.open(source, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"t": 100.0, "csi": packet}) + "\n")
                handle.write(json.dumps({"t": 100.1, "csi": packet}) + "\n")
            convert_wifi_presence_json(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 3, 30))
            self.assertTrue(np.allclose(loaded["timestamp_s"], [0, .1]))
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["label_sets"], {"Activity state": ["Mobile"]})

    def test_wifi_tad_official_loader_and_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "dataset"
            source = root / "smartwifi" / "validation_npy" / "video_01.npy"
            source.parent.mkdir(parents=True)
            np.save(source, np.full((100, 30), 40.0, dtype=np.float32))
            annotations = root / "annotations"
            annotations.mkdir()
            (annotations / "val_video_info.csv").write_text(
                "name,fps,sample_fps,count,sample_count\nvideo_01,30,10,300,100\n", encoding="utf-8"
            )
            (annotations / "val_Annotation_ours.csv").write_text(
                "name,x,class,start,end\nvideo_01,x,2,30,90\n", encoding="utf-8"
            )
            output = Path(directory) / "out.npz"
            convert_wifi_tad_npy(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (100, 1, 30))
            self.assertTrue(np.allclose(loaded["official_normalized_amplitude"], 1.0))
            self.assertTrue(np.allclose(loaded["segment_start_index"], [10]))
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["sample_rate_hz"], 100.0)
            self.assertEqual(sidecar["segments"][0]["label"], "walk")

    def test_operanet_official_mat_table_fields(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "exp001.mat", Path(directory) / "out.npz"
            table = {
                f"tx{tx}rx{rx}_sub{sub}": np.asarray([complex(sub, rx), complex(sub + 1, tx)])
                for tx in range(1, 4) for rx in range(1, 4) for sub in range(1, 31)
            }
            table.update(timestamp=np.asarray([1000, 1010]), activity=np.asarray(["walk", "sit"]),
                         person_id=np.asarray(["One", "One"]), room_no=np.asarray(["1", "1"]))
            savemat(source, {"wificsi": table})
            convert_operanet_mat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 30, 3, 3))
            self.assertTrue(np.allclose(loaded["timestamp_s"], [0, .01]))
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["axis_order"], ["time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(sidecar["sample_rate_hz"], 100.0)
            self.assertEqual(sidecar["nominal_sample_rate_hz"], 1600.0)
            self.assertIn("timestamp span", sidecar["sample_rate_provenance"])
            self.assertEqual(sidecar["label_sets"], {"Activity": ["sit", "walk"]})
            self.assertEqual(sidecar["setting_values"], {"Person": ["One"], "Room": ["1"]})

    def test_nist_breathesmart_official_real_imag_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_path = root / "config0001_csi_real_log.csv"
            imag_path = root / "config0001_csi_imag_log.csv"
            # Reproduce the official reserved envelope: three Rx groups with
            # three Tx slots each, while only Tx 1/2 and carrier 0..55 contain
            # measurements.
            real = np.zeros((4, 9, 114), dtype=np.float64)
            for rx in range(3):
                for tx in range(2):
                    real[:, rx * 3 + tx, :56] = 10 * (tx + 1) + rx
            np.savetxt(real_path, real.reshape(4, -1), delimiter=",")
            np.savetxt(imag_path, np.zeros((4, 1026)), delimiter=",")
            (root / "config0001.csv").write_text("bpm,15\nmsgFreq,10\n", encoding="utf-8")
            output = root / "out.npz"
            convert_nist_breathesmart(real_path, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (4, 56, 2, 3))
            self.assertTrue(np.all(loaded["amplitude"][:, :, 1, 2] == 22))
            self.assertEqual(loaded["source_subcarrier_mask"].tolist(), [True] * 56 + [False] * 58)
            self.assertEqual(
                loaded["source_link_mask"].tolist(),
                [True, True, False, True, True, False, True, True, False],
            )
            self.assertTrue(np.allclose(loaded["timestamp_s"], [0, .1, .2, .3]))
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["axis_order"], ["time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(sidecar["discarded_all_zero_subcarrier_slots"], 58)
            self.assertEqual(sidecar["discarded_all_zero_link_slots"], 3)

    def test_csida_official_zarr_arrays(self):
        import zarr
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shape = (3, 10, 3, 114)
            zarr.save(str(root / "csi_data_amp"), np.ones(shape, dtype=np.float32))
            zarr.save(str(root / "csi_data_pha"), np.zeros(shape, dtype=np.float32))
            for name, values in {
                "csi_label_act": [0, 1, 2], "csi_label_env": [0, 0, 1],
                "csi_label_loc": [0, 1, 2], "csi_label_user": [0, 1, 2],
            }.items():
                zarr.save(str(root / name), np.asarray(values))
            output = root / "out.npz"
            convert_csida_zarr(root / "csi_data_amp", output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (3, 10, 114, 1, 3))
            self.assertTrue(np.allclose(loaded["csi_real"], 1.0))
            self.assertEqual(loaded["subcarrier_index"].shape, (114,))
            self.assertEqual(loaded["tx_link_index"].tolist(), [0])
            self.assertEqual(loaded["rx_link_index"].tolist(), [0, 1, 2])
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(
                sidecar["axis_order"],
                ["sample", "time", "subcarrier", "tx_link", "rx_link"],
            )
            self.assertEqual(sidecar["sample_rate_hz"], 1000.0)

    def test_prepare_csida_resamples_time_and_preserves_native_signal_axes(self):
        import zarr
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "csida" / "original"
            original.mkdir(parents=True)
            source = np.arange(2 * 1800 * 3 * 114, dtype=np.float32).reshape(2, 1800, 3, 114)
            zarr.save(str(original / "csi_data_amp"), source)
            for name, values in {
                "csi_label_act": [0, 1], "csi_label_env": [0, 1],
                "csi_label_loc": [0, 2], "csi_label_user": [0, 4],
            }.items():
                zarr.save(str(original / name), np.asarray(values))

            summary = prepare_dataset("csida", root, setting="random")

            self.assertEqual(summary["view_options"]["target_rate_hz"], 100.0)
            self.assertEqual(summary["window_policy"]["selected_window_length"], 180)
            output = root / "csida" / "standardized" / "views" / "csi_data_amp.npz"
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (2, 180, 114, 1, 3))
                self.assertEqual(loaded["subcarrier_mask"].shape, (114,))
                self.assertEqual(loaded["tx_link_mask"].tolist(), [True])
                self.assertEqual(loaded["rx_link_mask"].tolist(), [True, True, True])
                self.assertEqual(loaded["activity_label"].tolist(), [0, 1])
            metadata = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["axis_order"],
                ["sample", "time", "subcarrier", "tx_link", "rx_link"],
            )
            self.assertFalse(metadata["target_rate_assumed"])

    def test_exposing_csi_official_ax_csi_preprocessing(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "S1_A.mat", Path(directory) / "out.npz"
            savemat(source, {"csi_buff": np.ones((8, 2048), dtype=np.complex64)})
            convert_exposing_csi_mat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 4, 1990))
            self.assertTrue(np.allclose(loaded["amplitude"], 1.0))
            self.assertEqual(str(loaded["source_label"]), "A")
            self.assertEqual(str(loaded["activity_label"]), "walk")
            sidecar = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["labels"]["activity"], "walk")

    def test_wiar_official_intel5300_binary_parser(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "csi_a1_1.dat", Path(directory) / "out.npz"
            expected_payload = (30 * (1 * 1 * 8 * 2 + 3) + 7) // 8
            records = []
            for timestamp in (1_000_000, 1_033_333):
                header = bytearray(20)
                header[0:4] = int(timestamp).to_bytes(4, "little")
                header[8], header[9] = 1, 1
                header[16:18] = expected_payload.to_bytes(2, "little")
                body = bytes(header) + bytes(expected_payload)
                records.append(struct.pack(">H", len(body) + 1) + bytes([187]) + body)
            source.write_bytes(b"".join(records))
            self.assertEqual(len(read_bf_file(source)), 2)
            convert_wiar_dat(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (2, 30, 1, 1))
            self.assertEqual(loaded["phase_rad"].shape, (2, 30, 1, 1))
            self.assertTrue(np.allclose(loaded["timestamp_s"], [0, .033333]))
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["label_sets"]["Activity"], ["horizontal arm wave"])
            self.assertEqual(sidecar["standard_representation"], "amplitude")
            self.assertAlmostEqual(sidecar["official_nominal_sample_rate_hz"], 30.0)
            self.assertAlmostEqual(sidecar["observed_timestamp_rate_hz"], 30.0003, places=3)

    def test_view_recomputes_wrapped_phase_from_resampled_iq(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "native.npz", Path(directory) / "view.npz"
            phase = np.asarray([3.0, -3.0], dtype=np.float32).reshape(2, 1, 1, 1)
            real, imag = np.cos(phase), np.sin(phase)
            np.savez_compressed(
                source, csi_real=real, csi_imag=imag,
                amplitude=np.ones_like(phase), phase_rad=phase,
            )
            source.with_suffix(".json").write_text(json.dumps({
                "standard_representation": "amplitude",
                "axis_order": ["time", "subcarrier", "tx_link", "rx_link"],
                "sample_rate_hz": 2.0,
            }))
            create_standard_view(
                source, output,
                ViewOptions(target_rate_hz=4.0, target_length=4),
            )
            with np.load(output) as loaded:
                interpolated = loaded["phase_rad"].reshape(-1)
            self.assertGreater(abs(float(interpolated[1])), 2.5)
            self.assertGreater(abs(float(interpolated[2])), 2.5)

    def test_wifi_80mhz_official_cfr_trace(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "AR1a_W.mat", Path(directory) / "out.npz"
            csi = np.arange(8 * 242, dtype=np.float32).reshape(8, 242).astype(np.complex64)
            savemat(source, {"csi_buff": csi})
            convert_wifi_80mhz_mat(source, output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (2, 4, 242))
                self.assertEqual(loaded["phase_rad"].shape, (2, 4, 242))
                self.assertTrue(np.array_equal(loaded["monitor_antenna_index"], [0, 1, 2, 3]))
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["source_shape"], [8, 242])
            self.assertEqual(sidecar["source_axis_order"], ["interleaved_packet_monitor_antenna", "subcarrier"])
            self.assertEqual(sidecar["label_sets"]["Activity"], ["walking"])
            self.assertEqual(sidecar["official_subset"], "AR1A")
            self.assertEqual(sidecar["antenna_layout"], {"tx_links": 1, "rx_links": 4})
            self.assertEqual(sidecar["sample_rate_hz"], 173.0)

    def test_wifi_80mhz_keeps_monitor_alignment_when_packet_is_zero(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "PC1a_n00.mat", Path(directory) / "out.npz"
            first = np.arange(4 * 242, dtype=np.float32).reshape(4, 242) + 1
            csi = np.concatenate([first, np.zeros((4, 242), dtype=np.float32)], axis=0).astype(np.complex64)
            savemat(source, {"csi_buff": csi})
            convert_wifi_80mhz_mat(source, output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (1, 4, 242))
                self.assertTrue(np.array_equal(loaded["amplitude"][0], first))
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["label_sets"]["Occupancy"], ["empty room"])

    def test_wifi_80mhz_canonicalizes_to_time_subcarrier_tx_rx(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "PI1a_p03.mat", Path(directory) / "out.npz"
            savemat(source, {"csi_buff": np.ones((12, 242), dtype=np.complex64)})
            convert_wifi_80mhz_mat(source, output)
            canonicalize_csi_output("wifi-80mhz", output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (3, 242, 1, 4))
                self.assertEqual(loaded["tx_link_index"].tolist(), [0])
                self.assertEqual(loaded["rx_link_index"].tolist(), [0, 1, 2, 3])
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["axis_order"], ["time", "subcarrier", "tx_link", "rx_link"])
            self.assertEqual(sidecar["label_sets"]["Identity"], ["p03"])
            self.assertEqual(sidecar["official_source_shape"], [12, 242])

    def test_wifi_80mhz_accepts_1024_bin_capture(self):
        from scipy.io import savemat
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "AR1a_W.mat", Path(directory) / "out.npz"
            savemat(source, {"csi_buff": np.ones((8, 1024), dtype=np.complex64)})
            convert_wifi_80mhz_mat(source, output)
            self.assertEqual(np.load(output)["amplitude"].shape, (2, 4, 242))

    def test_glasgow_usrp_amplitude_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "1_Subject_Sitting" / "1_Sitting_01.csv"
            source.parent.mkdir()
            np.savetxt(source, np.ones((52, 100)), delimiter=",")
            output = Path(directory) / "out.npz"
            convert_usrp_amplitude_csv("glasgow-multiuser", source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (100, 1, 52))
            self.assertEqual(str(loaded["source_label"]), "sitting")

    def test_glasgow_activity_localization_keeps_activity_and_location_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Location2_Z1Z2Z3" / "SittingL2Z3_5.csv"
            source.parent.mkdir()
            np.savetxt(source, np.ones((10, 52)), delimiter=",")
            output = Path(directory) / "out.npz"
            convert_usrp_amplitude_csv("glasgow-activity-localization", source, output)
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["label_sets"]["Activity"], ["sitting"])
            self.assertEqual(sidecar["label_sets"]["Location"], ["location 2, zone 3"])

    def test_wipe_fall_uses_official_risk_folder_and_51_subcarriers(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "high_unseen" / "trial.csv"
            source.parent.mkdir()
            np.savetxt(source, np.arange(8 * 51, dtype=np.float32).reshape(8, 51), delimiter=",")
            output = Path(directory) / "out.npz"
            convert_usrp_amplitude_csv("wipe-fall", source, output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (8, 51, 1, 1))
                self.assertEqual(str(loaded["activity_label"]), "high")
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["label_sets"]["Fall risk"], ["high"])
            self.assertEqual(sidecar["label_sets"]["Release split"], ["unseen test"])
            self.assertIsNone(sidecar["sample_rate_hz"])
            self.assertEqual(infer_metadata("high_unseen/trial.csv")["predefined_split"], "test")
            self.assertEqual(infer_metadata("med/trial.csv")["predefined_split"], "train")

    def test_prepare_wipe_fall_preserves_official_unseen_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "wipe-fall" / "original"
            for folder in ("low", "med", "high", "low_unseen", "med_unseen", "high_unseen"):
                path = original / folder / "trial.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savetxt(path, np.ones((12, 51), dtype=np.float32), delimiter=",")
            summary = prepare_dataset(
                "wipe-fall", root, setting="official_unseen",
                view_options=resolve_task_profile("general-sensing"),
            )
            self.assertEqual(summary["source_count"], 6)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(
                summary["split"]["partition_counts"], {"train": 3, "val": 0, "test": 3},
            )
            output = root / "wipe-fall" / "standardized" / "high_unseen__trial.npz"
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (12, 51, 1, 1))

    def test_wireless_har_selects_wifi_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "Wireless_sensing_human_activity_recognition" / "WiFi_CSI" / "Room_1" / "walk.csv"
            source.parent.mkdir(parents=True)
            np.savetxt(source, np.ones((12, 52)), delimiter=",")
            output = Path(directory) / "out.npz"
            convert_wireless_har_wifi(source, output)
            loaded = np.load(output)
            self.assertEqual(loaded["amplitude"].shape, (12, 1, 52))
            self.assertEqual(str(loaded["activity_label"]), "walk")

    def test_wireless_har_v73_uses_named_subcarrier_and_rx_axes(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            source = (
                Path(directory) / "Wireless_sensing_human_activity_recognition" /
                "WiFi_CSI" / "Room_2" / "sitrotate" / "sitrotate_1.mat"
            )
            source.parent.mkdir(parents=True)
            names = [
                f"tx1rx{rx}_sub{subcarrier}"
                for rx in range(1, 4)
                for subcarrier in range(1, 31)
            ]
            with h5py.File(source, "w") as handle:
                refs = handle.create_group("#refs#")
                name_refs = np.empty((90, 1), dtype=h5py.ref_dtype)
                value_refs = np.empty((90, 1), dtype=h5py.ref_dtype)
                complex_dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
                for index, name in enumerate(names):
                    chars = refs.create_dataset(
                        f"name_{index}",
                        data=np.asarray([ord(char) for char in name], dtype=np.uint16),
                    )
                    chars.attrs["MATLAB_class"] = np.bytes_(b"char")
                    values = np.zeros((1, 5), dtype=complex_dtype)
                    values["real"] = index
                    values["imag"] = -index
                    signal = refs.create_dataset(f"value_{index}", data=values)
                    signal.attrs["MATLAB_class"] = np.bytes_(b"double")
                    name_refs[index, 0] = chars.ref
                    value_refs[index, 0] = signal.ref
                names_cell = refs.create_dataset("variable_names", data=name_refs)
                names_cell.attrs["MATLAB_class"] = np.bytes_(b"cell")
                values_cell = refs.create_dataset("table_values", data=value_refs)
                values_cell.attrs["MATLAB_class"] = np.bytes_(b"cell")

            output = Path(directory) / "out.npz"
            convert_wireless_har_wifi(source, output)
            with np.load(output) as loaded:
                self.assertEqual(loaded["amplitude"].shape, (5, 30, 1, 3))
                self.assertEqual(str(loaded["activity_label"]), "sit and rotate")
                self.assertAlmostEqual(float(loaded["amplitude"][0, 0, 0, 2]), 60 * 2 ** 0.5)
            sidecar = json.loads(output.with_suffix(".json").read_text())
            self.assertEqual(sidecar["source_shape"], [5, 90])
            self.assertEqual(sidecar["antenna_layout"], {"tx_links": 1, "rx_links": 3})
            self.assertEqual(sidecar["label_sets"], {"Activity": ["sit and rotate"]})
            self.assertEqual(sidecar["settings"], {"room": "room 2"})

    def test_prepare_glassgow_release_without_rearranging_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "glasgow-multiuser" / "original" / "1_Subject_Sitting" / "1_Sitting_01.csv"
            source.parent.mkdir(parents=True)
            np.savetxt(source, np.ones((20, 52)), delimiter=",")
            summary = prepare_dataset("glasgow-multiuser", root, setting="random")
            self.assertEqual(summary["converted"], 1)
            self.assertTrue((root / "glasgow-multiuser" / "standardized" / "1_Subject_Sitting__1_Sitting_01.npz").exists())


if __name__ == "__main__":
    unittest.main()
