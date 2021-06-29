from typing import Any
import pytest
from portuguese_ner_biomedical.dataset import Data
from test_data import vocab_out_true

example_file: Any = "example_dir/"


class TestDataset:
    def test_constructor(self):
        data = Data(example_file)
        assert data.vocab_out == vocab_out_true
