# coding: utf-8
"""Tests for training a lemmatizer."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,redefined-outer-name
from typing import Tuple, List

import pytest

from lemmy.lemmatizer import _find_suffix_start, Lemmatizer, Token, Tag, Lemma
from utils import _prepare


@pytest.fixture(scope="module")
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


class TestTrainingDanish(object):
    """Test class for training a lemmatizer."""

    @pytest.mark.parametrize("train,test", [
        ([('noun', 'skaber', 'skaber'), ('noun', 'venskaber', 'venskab')],
         [('noun', 'skaber', ['skaber']), ('noun', 'venskaber', ['venskab'])]),

        ([('noun', 'skab', 'skab'), ('noun', 'skaber', 'skaber'), ('noun', 'venskaber', 'venskab')],
         [('noun', 'skab', ['skab']), ('noun', 'skaber', ['skaber']), ('noun', 'venskaber', ['venskab'])]),

        ([('noun', 'alen', 'alen'), ('noun', 'alen', 'ale')],
         [('noun', 'alen', ['alen', 'ale']), ('noun', 'alen', ['alen', 'ale'])]),

        ([('noun', 'alen', 'ale'), ('noun', 'alen', 'alen')],
         [('noun', 'alen', ['alen', 'ale']), ('noun', 'alen', ['alen', 'ale'])])
    ])
    def test_fit(self, lemmatizer, train, test):
        """Test training on small datasets."""
        X, y = _prepare(train)
        lemmatizer.fit(X, y)
        for word_class, full_form, expected_lemmas in test:
            actual_lemmas = lemmatizer.lemmatize(word_class, full_form, disambiguate=False)
            assert isinstance(actual_lemmas, list)
            assert len(actual_lemmas) == len(expected_lemmas)
            assert set(actual_lemmas) == set(expected_lemmas)

    @pytest.mark.parametrize("test_input,expected", [
        (("adelsmændene", "adelsmand", 4), 6),
        (("A'erne", "A", 1), 1)
    ])
    def test_find_suffix_start(self, test_input, expected):
        """Test splitting of full form to prefix and suffix."""
        full_form, lemma, min_rule_length = test_input
        actual = _find_suffix_start(full_form, lemma, min_rule_length)
        assert actual == expected


class TestTrainingHungarian(object):
    """Test class for training a lemmatizer."""

    @pytest.mark.parametrize("train,test", [
        ([('noun', 'alma', 'alom'), ('noun', 'alma', 'alma'), ('noun', 'alma', 'alma')],
         [('noun', 'alma', 'alma')]),
    ])
    def test_fit_predict(self, lemmatizer: Lemmatizer, train: List[Tuple[Tag, Token, Lemma]],
                         test: List[Tuple[Tag, Token, Lemma]]):
        """Test training on small datasets."""
        X, y = _prepare(train)
        lemmatizer.fit(X, y)
        for word_class, full_form, expected in test:
            lemma: Lemma = lemmatizer.lemmatize(word_class, full_form, disambiguate=True)
            assert lemma == expected
