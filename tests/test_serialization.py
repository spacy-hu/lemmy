# coding: utf-8
"""Tests for training a lemmatizer."""
# pylint: disable=protected-access,too-many-public-methods,no-self-use,too-few-public-methods,redefined-outer-name
from typing import Dict, Any

from lemmy.lemmatizer import Lemmatizer
from lemmy.serialization import Serializable
from utils import _prepare

TRAIN_DATA = [('noun', 'alma', 'alom'), ('noun', 'alma', 'alma'), ('noun', 'alma', 'alma')]
TEST_DATA = [("noun", "alma", "alma"), ("verb", "keres", "keres")]


def test_lemmatizer_serialization():
    """Test training on small datasets."""
    X, y = _prepare(TRAIN_DATA)
    lemmatizer = Lemmatizer()
    lemmatizer.fit(X, y)
    bytes_data: bytes = lemmatizer.to_bytes()
    restored_lemmatizer = Lemmatizer.from_bytes(bytes_data)

    assert lemmatizer._rule_repo == restored_lemmatizer._rule_repo
    assert dict(lemmatizer._lemma_counter._counts) == dict(restored_lemmatizer._lemma_counter._counts)

    for tag, word, lemma in TEST_DATA:
        lemma1 = lemmatizer.lemmatize(tag, word)
        lemma2 = restored_lemmatizer.lemmatize(tag, word)
        assert lemma2 == lemma1
        assert lemma2 == lemma


class DummyClass(Serializable["DummyClass"]):
    def __init__(self):
        self.value = 1

    def _to_bytes(self) -> Dict[str, Any]:
        return {"value": self.value}

    @classmethod
    def _from_bytes(cls, msg: Dict[str, Any]) -> "DummyClass":
        obj = DummyClass()
        obj.value = msg["value"]
        return obj

    @classmethod
    def _version(cls) -> int:
        pass


def test_dummy_serialization():
    obj1 = DummyClass()
    bytes_data = obj1.to_bytes()
    obj2 = DummyClass.from_bytes(bytes_data)
    assert obj1.value == obj2.value
