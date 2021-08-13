# coding: utf-8
"""A spaCy pipeline component."""
from pathlib import Path
from types import Union

from spacy.tokens.token import Token

from lemmy.lemmatizer import Lemmatizer


class LemmyPipelineComponent(object):
    """
    A pipeline component for spaCy.

    This wraps a trained lemmatizer for easy use with spaCy.
    """

    name = 'lemmy'

    def __init__(self, model_path: Union[str, Path]):
        """Initialize a pipeline component instance."""
        self._internal = Lemmatizer.from_disk(model_path)

    def __call__(self, doc):
        """
        Apply the pipeline component to a `Doc` object.

        doc (Doc): The `Doc` returned by the previous pipeline component.
        RETURNS (Doc): The modified `Doc` object.
        """
        token: Token
        for token in doc:
            token.lemma_ = self._internal.lemmatize(token.pos_, token.text)
        return doc


def load(model_path: Union[str, Path]):
    return LemmyPipelineComponent(model_path)
