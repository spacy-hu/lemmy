from lemmy import Lemmatizer


def load(language: str) -> Lemmatizer:
    """Load lemmatizer for specified language."""
    lookup = {'da': _load_da, 'sv': _load_sv}
    if language not in lookup:
        raise ValueError("Language not supported.")
    return lookup[language]()


def _load_da() -> Lemmatizer:
    from lemmy.rules.da import rules as da_rules
    return Lemmatizer(da_rules)


def _load_sv() -> Lemmatizer:
    from lemmy.rules.sv import rules as sv_rules
    return Lemmatizer(sv_rules)
