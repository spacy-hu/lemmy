import re
from pathlib import Path
from typing import Tuple, List

import conllu
import typer
from conllu import TokenList

from lemmy import Lemmatizer
from lemmy.lemmatizer import Lemma, Token, Tag

app = typer.Typer()

TaggedWords = List[Tuple[Tag, Token]]
Lemmata = List[Lemma]

CONTRACTION_PATTERN = re.compile(r"(\w)(\+)(\w)")


def read_file(path: Path, ignore_contractions: bool = True) -> Tuple[TaggedWords, Lemmata]:
    with path.open() as f:
        sentences: List[TokenList] = conllu.parse(f.read().strip())

        X = [
            (token["upos"], token["form"])
            for sentence in sentences
            for token in sentence
        ]
        y = [token["lemma"] for sentence in sentences for token in sentence]
        if ignore_contractions:
            y = [CONTRACTION_PATTERN.sub(r"\1\3", lemma) for lemma in y]
        return X, y


@app.command()
def debug(model_path: Path, test_data: Path, ignore_contractions: bool = True):
    lemmatizer: Lemmatizer = Lemmatizer.from_disk(model_path)
    tagged_words, lemmata = read_file(test_data, ignore_contractions)
    predicted: List[Lemma] = [lemmatizer.lemmatize(tag, word) for tag, word in tagged_words]
    for lemma, pred, (tag, word) in zip(lemmata, predicted, tagged_words):
        if lemma != pred:
            print(f"Wrong lemma for {word}[{tag}]: '{pred}', should be '{lemma}'")


@app.command()
def evaluate(model_path: Path, test_data: Path, ignore_contractions: bool = True):
    lemmatizer: Lemmatizer = Lemmatizer.from_disk(model_path)
    tagged_words, lemmata = read_file(test_data, ignore_contractions)
    predicted: List[Lemma] = [lemmatizer.lemmatize(tag, word) for tag, word in tagged_words]
    accuracy = sum([gt == pred for gt, pred in zip(lemmata, predicted)]) / float(len(lemmata))
    print(f"Accuracy: {accuracy:.2%}")


@app.command()
def train(train_path: Path, model_path: Path, max_iterations: int = typer.Option(1, "--max-iter", "-m"),
          ignore_contractions: bool = True):
    model_path.parent.mkdir(parents=True, exist_ok=True)

    tagged_words, lemmata = read_file(train_path, ignore_contractions)
    lemmatizer: Lemmatizer = Lemmatizer()
    lemmatizer.fit(tagged_words, lemmata, max_iteration=max_iterations)

    predicted: List[Lemma] = [lemmatizer.lemmatize(tag, word) for tag, word in tagged_words]
    accuracy = sum([gt == pred for gt, pred in zip(lemmata, predicted)]) / float(len(lemmata))
    print(f"Accuracy: {accuracy:.2%}")

    lemmatizer.to_disk(model_path)


if __name__ == "__main__":
    app()
