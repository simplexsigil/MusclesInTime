import unittest
from musint.datasets.mint_dataset import MintDataset
import argparse

class TestMintDataset(unittest.TestCase):
    dataset_path = ""

    @classmethod
    def setUpClass(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset_path', required=True)
        args, unknown = parser.parse_known_args()
        cls.dataset_path = args.dataset_path

    def setUp(self):
        self.mint_dataset = MintDataset(self.__class__.dataset_path)

    def test_getitem(self):
        sample = self.mint_dataset[0]
        self.assertIsNotNone(sample)

    def test_by_path_id(self):
        sample_by_path_id = self.mint_dataset.by_path_id("s1/acting2")
        self.assert_sample_example(sample_by_path_id)

    def test_by_path(self):
        sample_by_path = self.mint_dataset.by_path("TotalCapture/TotalCapture/s1/acting2_poses")
        self.assert_sample_example(sample_by_path)

    def test_by_babel_sid(self):
        sample_by_babel_sid = self.mint_dataset.by_babel_sid(12906)
        self.assert_sample_example(sample_by_babel_sid)

    def test_by_subject_and_sequence(self):
        sample_by_subject_and_sequence = self.mint_dataset.by_subject_and_sequence("s1", "acting2_poses")
        self.assert_sample_example(sample_by_subject_and_sequence)

    def test_by_humanml3d_name(self):
        sample_by_humanml3d_name = self.mint_dataset.by_humanml3d_name("002418")[0]
        self.assert_sample_example(sample_by_humanml3d_name)
        sample_by_humanml3d_name = self.mint_dataset.by_humanml3d_name("M002418")[0]
        self.assert_sample_example(sample_by_humanml3d_name)
        sample_by_humanml3d_name = self.mint_dataset.by_humanml3d_name("002418.npy")[0]
        self.assert_sample_example(sample_by_humanml3d_name)

    def test_get_forces_20fps(self):
        sample = self.mint_dataset[0]
        forces = sample.get_forces((0.5, 0.6), 20.0)
        assert forces.index[0] == 0.5
        assert forces.index[1] == 0.55
        assert forces.index[2] == 0.6
        assert len(forces.index) == 3

    def test_get_forces_50fps(self):
        sample = self.mint_dataset[0]
        forces = sample.get_forces((0.5, 0.6), 50.0)
        assert forces.index[0] == 0.5
        assert forces.index[1] == 0.52
        assert forces.index[2] == 0.54
        assert forces.index[3] == 0.56
        assert forces.index[4] == 0.58
        assert forces.index[5] == 0.6
        assert len(forces.index) == 6
    
    def test_get_grf_20fps(self):
        sample = self.mint_dataset[0]
        grf = sample.get_grf((0.5, 0.6), 20.0)
        assert grf.index[0] == 0.5
        assert grf.index[1] == 0.55
        assert grf.index[2] == 0.6
        assert len(grf.index) == 3

    def test_get_grf_50fps(self):
        sample = self.mint_dataset[0]
        grf = sample.get_grf((0.5, 0.6), 50.0)
        assert grf.index[0] == 0.5
        assert grf.index[1] == 0.52
        assert grf.index[2] == 0.54
        assert grf.index[3] == 0.56
        assert grf.index[4] == 0.58
        assert grf.index[5] == 0.6
        assert len(grf.index) == 6
    
    def test_get_muscle_activations_20fps(self):
        sample = self.mint_dataset[0]
        muscle_activations = sample.get_muscle_activations((0.5, 0.6), 20.0)
        assert muscle_activations.index[0] == 0.5
        assert muscle_activations.index[1] == 0.55
        assert muscle_activations.index[2] == 0.6
        assert len(muscle_activations.index) == 3
    
    def test_get_muscle_activations_50fps(self):
        sample = self.mint_dataset[0]
        muscle_activations = sample.get_muscle_activations((0.5, 0.6), 50.0)
        assert muscle_activations.index[0] == 0.5
        assert muscle_activations.index[1] == 0.52
        assert muscle_activations.index[2] == 0.54
        assert muscle_activations.index[3] == 0.56
        assert muscle_activations.index[4] == 0.58
        assert muscle_activations.index[5] == 0.6
        assert len(muscle_activations.index) == 6

    def test_get_valid_indices_20fps(self):
        sample = self.mint_dataset[0]
        valid_indices = sample.get_valid_indices((0.5, 0.6), 20.0)
        assert valid_indices[0] == 0.5
        assert valid_indices[1] == 0.55
        assert valid_indices[2] == 0.6
        assert len(valid_indices) == 3

    def test_get_valid_indices_50fps(self):
        sample = self.mint_dataset[0]
        valid_indices = sample.get_valid_indices((0.5, 0.6), 50.0)
        assert valid_indices[0] == 0.5
        assert valid_indices[1] == 0.52
        assert valid_indices[2] == 0.54
        assert valid_indices[3] == 0.56
        assert valid_indices[4] == 0.58
        assert valid_indices[5] == 0.6
        assert len(valid_indices) == 6

    def test_get_valid_indices_by_frame_20fps(self):
        sample = self.mint_dataset[0]
        valid_indices = sample.get_valid_indices((0.5, 0.6), 20.0, as_time=False)
        assert valid_indices[0] == 10
        assert valid_indices[1] == 11
        assert valid_indices[2] == 12
        assert len(valid_indices) == 3

    def test_get_valid_indices_by_frame_50fps(self):
        sample = self.mint_dataset[0]
        valid_indices = sample.get_valid_indices((0.5, 0.6), 50.0, as_time=False)
        assert valid_indices[0] == 25
        assert valid_indices[1] == 26
        assert valid_indices[2] == 27
        assert valid_indices[3] == 28
        assert valid_indices[4] == 29
        assert valid_indices[5] == 30
        assert len(valid_indices) == 6

    def test_get_gaps(self):
        for i in range(100):
            sample = self.mint_dataset[i]
            gaps = sample.get_gaps()
            if sample.has_gap:
                assert len(gaps) > 0
                assert gaps[0][0] < gaps[0][1]
                assert gaps[0][0] + gaps[0][1] > 1
            else:
                self.assertListEqual(gaps, [])

    def test_get_gaps_by_frame(self):
        for i in range(100):
            sample = self.mint_dataset[i]
            gaps = sample.get_gaps(as_frame=True)
            if sample.has_gap:
                assert len(gaps) > 0
                assert gaps[0][0] < gaps[0][1]
                assert gaps[0][0] + gaps[0][1] > 20
            else:
                self.assertListEqual(gaps, [])

    def assert_sample_example(self, sample_by_subject_and_sequence):
        assert sample_by_subject_and_sequence.babel_sid == 12906
        assert sample_by_subject_and_sequence.subject == "s1"
        assert sample_by_subject_and_sequence.sequence == "acting2_poses"
        assert sample_by_subject_and_sequence.dataset == "TotalCapture"
        assert sample_by_subject_and_sequence.path_id == "s1/acting2"
        assert sample_by_subject_and_sequence.data_path == "TotalCapture/TotalCapture/s1/acting2_poses"
        assert not sample_by_subject_and_sequence.has_gap
        assert "002418.npy" in sample_by_subject_and_sequence.humanml3d_name
        assert "003266.npy" in sample_by_subject_and_sequence.humanml3d_name

if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)