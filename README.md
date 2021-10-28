# 🤘 Lemmy3

Lemmy3 is an experimental fork of [Lemmy](https://github.com/sorenlind/lemmy)

It has been refactored in object-oriented manner, the codebase is extended with type-hints and spacy-compatible serialization, and a simple frequency-based disambiguation method is added. The tool comes with a command line interface (`lemmy`) for training lemmatizer models.

## Installation

To get started using the lemmatizer, install it from PyPI:

```bash
pip install lemmy3
```

## Usage

To train a model one can use the packaged CLI. 
(Note that your data must be in [CONLLU format](https://universaldependencies.org/format.html))

```bash
lemmy train train.conllu model.bin
```

Evaluation of the model is simply possible by issuing the following command.

```bash
lemmy evaluate model.bin test.conllu
```

### Using `lemmy3` with spaCy

```python
import lemmy

nlp.add_pipe("lemmy3", before="parser")
```
