"""Initialization code for Lemmy."""
from pathlib import Path
from typing import Union

from spacy import Language

from lemmy.lemmatizer import Lemmatizer


@Language.factory("lemmy3", default_config={"data": None})
def create(nlp: Language, name: str, data: Union[str, Path]) -> Lemmatizer:
    return Lemmatizer
