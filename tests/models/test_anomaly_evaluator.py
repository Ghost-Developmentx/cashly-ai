import numpy as np

from app.models.anomaly.evaluator import AnomalyEvaluator


def test_evaluate_returns_expected_keys():
    evaluator = AnomalyEvaluator()
    y_true = np.array([1, -1, 1, -1, -1])
    y_pred = np.array([1, -1, 1, 1, -1])

    metrics = evaluator.evaluate(y_true, y_pred)
    expected = {
        'precision',
        'recall',
        'f1_score',
        'accuracy',
        'specificity',
        'true_positive_rate',
        'false_positive_rate',
        'true_positives',
        'false_positives',
        'false_negatives',
        'true_negatives',
    }
    for key in expected:
        assert key in metrics


def test_evaluate_with_scores_returns_expected_keys():
    evaluator = AnomalyEvaluator()
    y_true = np.array([1, -1, 1, -1, -1])
    scores = np.array([0.1, 0.9, 0.2, 0.8, 0.7])

    metrics = evaluator.evaluate_with_scores(y_true, scores)
    expected = {'auc_roc', 'optimal_threshold', 'optimal_f1'}
    for key in expected:
        assert key in metrics