from contextlib import contextmanager

import pandas as pd
import pytest
from dsxt.threshold import (ThresholdClassifier,
                            threshold_of_best_recall_at_precison)
from numpy.testing import assert_array_equal
from sklearn.base import BaseEstimator, ClassifierMixin


@contextmanager
def does_not_raise():
    yield


@pytest.mark.parametrize(
    'y_true, probas_pred, precision_constraint, expected_result, expected_exception',
    [
        ([1, 0], [0.4, 0.2], 1, 0.4, does_not_raise()),
        ([1, 1, 0], [0.7, 0.4, 0.2], 1, 0.4, does_not_raise()),
        ([1, 1, 0], [0.7, 0.2, 0.4], 1, 0.7, does_not_raise()),
        ([1, 1, 0], [0.7, 0.2, 0.4], 0.66, 0.2, does_not_raise()),
        ([1, 1, 0], [0.7, 0.2, 0.4], 0.67, 0.7, does_not_raise()),
        ([1, 0, 1], [0.7, 0.7, 0.7], 1, [], does_not_raise())
    ]
)
def test_threshold_of_best_recall_at_precison(y_true, probas_pred, precision_constraint, expected_result, expected_exception):
    with expected_exception:
        assert_array_equal(threshold_of_best_recall_at_precison(y_true, probas_pred, precision_constraint), expected_result)


class DummyClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, classes, class_probas):
        self.classes_ = classes
        self.class_probas = class_probas

    def fit(self, X, y):
        pass

    def predict_proba(self, X):
        return self.class_probas


@pytest.mark.parametrize(
    'classes, class_probas, thresholds, fallback_class, expected_result',
    [
        (
            ['class_a', 'class_b'],
            [[0.8, 0.2]],
            [0.5, 0.4],
            '',
            ['class_a']
        ),
        (
            ['class_a', 'class_b'],
            [[0.4, 0.6]],
            [0.5, 0.4],
            '',
            ['class_b']
        ),
        (
            ['class_a', 'class_b'],
            [[0.4, 0.6]],
            [0.3, 0.7],
            '',
            ['class_a']
        ),
        (
            ['class_a', 'class_b'],
            [[0, 0]],
            [0.5, 0.5],
            'other',
            ['other']
        ),
        (
            ['class_a', 'class_b'],
            [[0.5, 0.5]],
            [0.5, 0.5],
            '',
            ['class_a']  # return first class if probas are equal
        ),
        (
            ['class_a', 'class_b'],
            [
                [0.6, 0.4],
                [0.4, 0.6],
                [0, 0]
            ],
            [0.5, 0.5],
            'other',
            ['class_a', 'class_b', 'other']
        ),
    ]
)
def test_threshold_classifier_predict(classes, class_probas, thresholds, fallback_class, expected_result):
    threshold_clf = ThresholdClassifier(
        estimator=DummyClassifier(classes, class_probas),
        thresholds=thresholds,
        fallback_class=fallback_class
    )
    class_labels = threshold_clf.predict(None)
    print(class_labels)
    assert len(class_labels.compare(pd.Series(expected_result))) == 0
