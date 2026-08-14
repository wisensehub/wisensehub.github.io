import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.build_previews import (
    canonical_csi_tensor,
    clean_dataset_label,
    flattened_source_link_tensor,
    mimo_link_stacks,
    label_relative_dir,
    preview_color_limits,
    raw_aril,
    raw_csida_rx_grids,
    raw_nist_stored_tensor,
    raw_ntu_fi_tensor,
    standardized_csida_rx_grids,
    split_catalog_key,
)


class BuildPreviewTests(unittest.TestCase):
    def test_flattened_source_link_tensor_keeps_all_four_links_once(self):
        value = np.arange(5 * 7 * 2 * 2).reshape(5, 7, 2, 2)
        flattened = flattened_source_link_tensor(value)
        self.assertEqual(flattened.shape, (5, 7, 1, 4))
        np.testing.assert_array_equal(flattened[:, :, 0], value.reshape(5, 7, 4))

    def test_flattened_source_link_tensor_keeps_all_nine_links_once(self):
        value = np.arange(5 * 7 * 3 * 3).reshape(5, 7, 3, 3)
        flattened = flattened_source_link_tensor(value)
        self.assertEqual(flattened.shape, (5, 7, 1, 9))
        np.testing.assert_array_equal(flattened[:, :, 0], value.reshape(5, 7, 9))

    def test_aril_before_preview_keeps_one_native_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "train_data_split_amp.mat"
            savemat(source, {"train_data": np.zeros((16, 52, 192), dtype=np.float32)})

            matrix, sample_rate = raw_aril(source)

            self.assertEqual(matrix.shape, (52, 192))
            self.assertIsNone(sample_rate)

    def test_csida_preview_keeps_three_receivers_separate(self):
        import json
        import zarr

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = np.zeros((2, 20, 3, 114), dtype=np.float32)
            source[:, :, 0, :] = 1
            source[:, :, 1, :] = 2
            source[:, :, 2, :] = 3
            zarr.save(str(root / "csi_data_amp"), source)

            raw, raw_rate = raw_csida_rx_grids(root / "csi_data_amp")
            self.assertEqual(raw.shape, (3, 114, 20))
            self.assertEqual(raw_rate, 1000.0)
            self.assertTrue(np.all(raw[0] == 1))
            self.assertTrue(np.all(raw[2] == 3))

            canonical = source.transpose(0, 1, 3, 2)[:, :, :, None, :]
            output = root / "standardized.npz"
            np.savez_compressed(output, amplitude=canonical)
            output.with_suffix(".json").write_text(json.dumps({
                "axis_order": ["sample", "time", "subcarrier", "tx_link", "rx_link"],
                "standard_representation": "amplitude",
                "sample_rate_hz": 100.0,
            }), encoding="utf-8")
            standardized, rate = standardized_csida_rx_grids(output, output.with_suffix(".json"))
            self.assertEqual(standardized.shape, (3, 114, 20))
            self.assertEqual(rate, 100.0)
            self.assertTrue(np.all(standardized[1] == 2))

    def test_nist_before_preview_keeps_reserved_storage_slots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real_path = root / "config0001_csi_real_log.csv"
            stored = np.zeros((4, 9, 114), dtype=np.float64)
            stored[:, [0, 1, 3, 4, 6, 7], :56] = 1.0
            np.savetxt(real_path, stored.reshape(4, -1), delimiter=",")
            np.savetxt(
                root / "config0001_csi_imag_log.csv",
                np.zeros((4, 1026), dtype=np.float64),
                delimiter=",",
            )

            tensor = raw_nist_stored_tensor(real_path)

            self.assertEqual(tensor.shape, (4, 114, 1, 9))
            self.assertTrue(np.all(tensor[:, :56, 0, 0] == 1))
            self.assertTrue(np.all(tensor[:, 56:, 0, :] == 0))
            self.assertTrue(np.all(tensor[:, :, 0, 2] == 0))

    def test_ntu_fi_before_preview_keeps_all_source_packets(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "walk.mat"
            value = np.arange(3 * 114 * 20, dtype=np.float32).reshape(3, 114, 20)
            savemat(source, {"CSIamp": value})

            tensor = raw_ntu_fi_tensor(source)

            self.assertEqual(tensor.shape, (20, 114, 1, 3))
            np.testing.assert_array_equal(tensor[:, :, 0, 1], value[1].T)

    def test_canonical_preview_keeps_both_antenna_axes(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.arange(12 * 30 * 3 * 3, dtype=np.float32).reshape(12, 30, 3, 3)
            output = root / "mimo.npz"
            np.savez_compressed(output, amplitude=value)
            output.with_suffix(".json").write_text(json.dumps({
                "axis_order": ["time", "subcarrier", "tx_link", "rx_link"],
                "standard_representation": "amplitude",
                "sample_rate_hz": 100.0,
            }), encoding="utf-8")

            loaded, rate = canonical_csi_tensor(output, output.with_suffix(".json"))
            self.assertEqual(loaded.shape, (12, 30, 3, 3))
            self.assertEqual(rate, 100.0)
            self.assertTrue(np.array_equal(loaded[:, :, 2, 1], value[:, :, 2, 1]))

    def test_canonical_preview_preserves_signed_processed_values(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.linspace(-2, 2, 12 * 3, dtype=np.float32).reshape(12, 3, 1, 1)
            output = root / "signed.npz"
            np.savez_compressed(output, amplitude=value)
            output.with_suffix(".json").write_text(json.dumps({
                "axis_order": ["time", "subcarrier", "tx_link", "rx_link"],
                "standard_representation": "amplitude",
                "source_representation": "processed_amplitude",
            }), encoding="utf-8")
            loaded, _ = canonical_csi_tensor(output, output.with_suffix(".json"))
            self.assertLess(float(loaded.min()), 0)
            self.assertTrue(np.array_equal(loaded, value))

    def test_mimo_stacks_include_every_tx_rx_pair(self):
        value = np.zeros((2, 3, 3, 3), dtype=np.float32)
        for tx in range(3):
            for rx in range(3):
                value[:, :, tx, rx] = tx * 10 + rx
        tx_stack, tx_labels, rx_stack, rx_labels = mimo_link_stacks(value)
        self.assertEqual(tx_stack.shape[0], 9)
        self.assertEqual(rx_stack.shape[0], 9)
        self.assertEqual(tx_labels, [f"T{tx}R{rx}" for tx in range(1, 4) for rx in range(1, 4)])
        self.assertEqual(rx_labels, [f"R{rx}T{tx}" for rx in range(1, 4) for tx in range(1, 4)])
        self.assertTrue(np.all(tx_stack[8] == 22))

    def test_label_cleanup_rejects_padding_and_maps_codes(self):
        self.assertIsNone(clean_dataset_label("figshare-csi-har", "  "))
        self.assertEqual(clean_dataset_label("operanet", "sit "), "sit")
        self.assertEqual(clean_dataset_label("wifi-80mhz", "W"), "walking")

    def test_task_separator_does_not_split_slash_inside_label(self):
        key = "Activity/sit down / get up"
        self.assertEqual(split_catalog_key(key), ("Activity", "sit down / get up"))
        self.assertEqual(label_relative_dir(key), Path("activity/sit_down_get_up"))

    def test_unsigned_preview_limits_ignore_isolated_spikes(self):
        values = np.concatenate([np.linspace(100, 900, 990), np.full(10, 10277.0)])
        low, high = preview_color_limits(values)
        self.assertGreaterEqual(low, 100)
        self.assertLess(high, 2000)


if __name__ == "__main__":
    unittest.main()
