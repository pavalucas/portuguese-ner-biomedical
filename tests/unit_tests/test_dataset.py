from typing import Any
from portuguese_ner_biomedical.dataset import Data
import test_data
import torch

example_file: Any = "example_dir/"


class TestDataset:
    def test_constructor(self):
        data = Data(example_file)
        assert data.vocab_out == test_data.vocab_out_true
        assert len(data.vocab_in) > 0
        assert len(data.corpus) > 0
        assert len(data.in_w2id) > 0
        assert len(data.in_id2w) > 0
        assert len(data.out_w2id) > 0
        assert len(data.out_id2w) > 0

    def test_preprocess(self):
        data = Data(example_file)
        token_ids, tag_ids = data.preprocess(test_data.example_tokens, test_data.example_tags)
        assert len(token_ids) > 0
        assert len(tag_ids) > 0
        assert type(token_ids) == torch.Tensor
        assert type(tag_ids) == torch.Tensor
