import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.fetch_samples import (
    _operanet_v73_csi_columns,
    select_aril_representatives,
    select_csida_representatives,
)


class FetchSampleTests(unittest.TestCase):
    def test_aril_subset_covers_every_location_and_gesture(self):
        activity = np.tile(np.arange(6), 16)
        location = np.repeat(np.arange(16), 6)

        selected = select_aril_representatives(activity, location)

        self.assertEqual(len(selected), 16)
        self.assertEqual(set(location[selected].tolist()), set(range(16)))
        self.assertEqual(set(activity[selected].tolist()), set(range(6)))

    def test_csida_subset_covers_every_released_label_value(self):
        labels = {
            "csi_label_act": np.tile(np.arange(6), 10),
            "csi_label_env": np.tile(np.arange(2), 30),
            "csi_label_loc": np.tile(np.arange(3), 20),
            "csi_label_user": np.tile(np.arange(5), 12),
        }

        selected = select_csida_representatives(labels)

        for name, values in labels.items():
            self.assertEqual(set(values[selected].tolist()), set(values.tolist()))

    def test_operanet_v73_reader_keeps_real_complex_values(self):
        import h5py

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "official.mat"
            names = [
                f"tx{tx}rx{rx}_sub{sub}"
                for tx in range(1, 4)
                for rx in range(1, 4)
                for sub in range(1, 31)
            ]
            with h5py.File(source, "w") as handle:
                refs = handle.create_group("#refs#")
                name_refs = np.empty((1, len(names)), dtype=h5py.ref_dtype)
                value_refs = np.empty((1, len(names)), dtype=h5py.ref_dtype)
                complex_dtype = np.dtype([("real", "<f8"), ("imag", "<f8")])
                for index, name in enumerate(names):
                    chars = refs.create_dataset(
                        f"name_{index}", data=np.asarray([ord(char) for char in name], dtype=np.uint16)
                    )
                    values = np.zeros((1, 4), dtype=complex_dtype)
                    values["real"] = index + np.arange(4)
                    values["imag"] = -index - np.arange(4)
                    signal = refs.create_dataset(f"value_{index}", data=values)
                    name_refs[0, index] = chars.ref
                    value_refs[0, index] = signal.ref
                names_cell = refs.create_dataset("variable_names", data=name_refs)
                names_cell.attrs["MATLAB_class"] = np.bytes_(b"cell")
                values_cell = refs.create_dataset("table_values", data=value_refs)
                values_cell.attrs["MATLAB_class"] = np.bytes_(b"cell")

            columns = _operanet_v73_csi_columns(source, max_rows=3)
            self.assertEqual(len(columns), 270)
            self.assertEqual(columns["tx1rx1_sub1"].shape, (3,))
            self.assertTrue(np.allclose(columns["tx1rx1_sub2"], [1 - 1j, 2 - 2j, 3 - 3j]))


if __name__ == "__main__":
    unittest.main()
