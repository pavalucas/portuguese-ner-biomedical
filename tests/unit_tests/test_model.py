from typing import Any
from portuguese_ner_biomedical.model import BERTSlotFilling, CRF, LinearLayerCRF, SimpleLinear
import test_data
import torchcrf
import sklearn_crfsuite
import torch.nn as nn


class TestModel:
    def test_BERTSlotFilling(self):
        model = BERTSlotFilling(test_data.hidden_dim, test_data.num_classes)
        assert model.hidden_dim == test_data.hidden_dim
        assert model.num_classes == test_data.num_classes

    def test_CRF(self):
        model = CRF()
        assert type(model.crf) == sklearn_crfsuite.CRF

    def test_LinearLayerCRF(self):
        model = LinearLayerCRF(test_data.num_classes, test_data.vocab_size, test_data.out_w2id)
        assert type(model.crf) == torchcrf.CRF

    def test_SimpleLinear(self):
        model = SimpleLinear(test_data.vocab_size, test_data.num_classes, test_data.out_w2id)
        assert type(model.emb) == nn.Embedding
        assert type(model.emb2tag) == nn.Linear
