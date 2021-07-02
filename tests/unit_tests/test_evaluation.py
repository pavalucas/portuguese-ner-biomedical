from portuguese_ner_biomedical.evaluation import Evaluation
import test_data


class TestEvaluation:
    def test_evaluation(self):
        evaluation = Evaluation(test_data.output_folder)
        assert evaluation.output_folder == test_data.output_folder
