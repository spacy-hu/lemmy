import pytest

from lemmy import Lemmatizer


@pytest.fixture(scope="module")
def lemmatizer() -> Lemmatizer:
    return Lemmatizer()


def _prepare(data):
    X = []
    y = []
    for word_class, full_form, lemma in data:
        X += [(word_class, full_form)]
        y += [lemma]
    return X, y
