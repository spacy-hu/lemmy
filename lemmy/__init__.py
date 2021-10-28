"""Initialization code for Lemmy."""
from pathlib import Path
from typing import Union

from spacy import Language

from lemmy.lemmatizer import Lemmatizer


# noinspection PyUnusedLocal
@Language.factory("lemmy3")
def create(nlp: Language, name: str, model_path: Union[str, Path]) -> "HunLemmatizer":
    return Lemmatizer.from_disk(model_path)
