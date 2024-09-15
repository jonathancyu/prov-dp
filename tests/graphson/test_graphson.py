import json
from src.graphson.raw_graph import RawGraph


class TestGraphson:
    def test_preserves_values(self):
        with open("./tests/graphson/test.json", "r") as f:
            input_dict = json.load(f)
            graph = RawGraph(**input_dict)

        with open("./tests/graphson/test_output.json", "w") as f:
            output_dict = graph.to_dict()
            json.dump(output_dict, f)
        assert input_dict == output_dict
