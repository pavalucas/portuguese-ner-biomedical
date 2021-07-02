from portuguese_ner_biomedical.trainer import Trainer
import test_data


class TestTrainer:
    def test_trainer(self):
        trainer = Trainer(test_data.model, test_data.batch)
        trainer.model.train()
        trainer.model.eval()
        assert trainer.model == test_data.model
