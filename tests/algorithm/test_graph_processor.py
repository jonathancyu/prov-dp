from argparse import Namespace
from pathlib import Path
from src.cli.perturb import run_processor


class TestRunConfigurations:
    def test_tc3_theia(self):
        data_path = (
            Path.home() / "workspace" / "SyssecLab" / "differential-privacy" / "data"
        )
        output_path = data_path / "output"
        args = {
            "input_dir": data_path / "benign_graphs" / "tc3-theia" / "firefox" / "nd",
            "output_dir": output_path / "tc3-theia" / "data2" / "benign",
            "reattach_mode": "bucket",
            "num_graphs": 10,
            "epsilon_1": 0.1,
            "epsilon_2": 0.1,
            "alpha": 0.5,
            "beta": 0.5,
            "gamma": 0.5,
            "num_epochs": 100,
            "prediction_batch_size": 5,
            "single_threaded": True
        }
        arg_namespace = Namespace(**args)
        run_processor(arg_namespace)
