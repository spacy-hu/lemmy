# coding: utf-8
"""Functions for lemmatizing using a set of lemmatization rules."""
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Tuple, List, Dict, Optional, TypeVar, Generic, Union, Any, NamedTuple

import spacy

from lemmy.serialization import Serializable, C

Tag = str
Lemma = str
Token = str
Suffix = str
Position = int


class SuffixWithEndMarker(NamedTuple):
    suffix: Suffix
    end_marker: bool


@dataclass
class TokenTransformations:
    token_suffix: Suffix
    lemma_suffixes: List[SuffixWithEndMarker] = field(default_factory=list)

    @classmethod
    def create(cls, token: Token, lemma: Lemma, current_rule_length: int) -> "TokenTransformations":
        if current_rule_length >= len(token) - 1:
            # The current longest matching rule is at least as long as the full form minus one character. Thus, building
            # a longer rule will use the entire full form.
            token_suffix: Suffix
            lemma_suffix: Suffix
            exhausted: bool
            token_suffix, lemma_suffix, exhausted = token, lemma, True
            return TokenTransformations(token_suffix, [SuffixWithEndMarker(lemma_suffix, exhausted)])

        min_rule_length: int = current_rule_length + 1
        split_position: int = _find_suffix_start(token, lemma, min_rule_length)
        token_suffix: Suffix = token[split_position:]
        lemma_suffix: Suffix = lemma[split_position:]
        assert min_rule_length <= len(token_suffix)

        exhausted = len(token_suffix) == len(token)
        return TokenTransformations(token_suffix, [SuffixWithEndMarker(lemma_suffix, exhausted)])

    def __call__(self, token: Token) -> List[Lemma]:
        """
        Apply specified rule to specified full form.

        Replace the the part of specified full form that matches the rule and replace with the lemma suffix of the rule.
        """
        if self.token_suffix == "":
            return [token + lemma_suffix.suffix for lemma_suffix in self.lemma_suffixes]

        prefix: str = token.rpartition(self.token_suffix)[0]
        return [prefix + lemma_suffix.suffix for lemma_suffix in self.lemma_suffixes]


class RuleRepository(Dict[Tag, Dict[Suffix, List[SuffixWithEndMarker]]], Serializable["RuleRepository"]):
    @classmethod
    def _version(cls) -> int:
        return 1

    def _to_bytes(self) -> Dict[str, Any]:
        return dict(self)

    @classmethod
    def _from_bytes(cls, msg: Dict[str, Any]) -> C:
        return RuleRepository(msg)

    def __init__(self, value: Dict = None):
        if value is None:
            value = {}
        super().__init__()
        tag: Tag
        for tag, inner_dict in value.items():
            self[tag] = {
                token_suffix: [SuffixWithEndMarker(lemma_suffix, end_marker)
                               for lemma_suffix, end_marker in lemma_suffixes_w_end_markers]
                for token_suffix, lemma_suffixes_w_end_markers in inner_dict.items()
            }

    def __missing__(self, key: Tag):
        self[key] = defaultdict(list)
        return self[key]

    def get_longest_matching_rule_for(self, tag: Tag, token: Token) -> TokenTransformations:
        """Find the rule with the longest full form suffix matching specified full form and class."""
        best: TokenTransformations = TokenTransformations("", [SuffixWithEndMarker("", True)])
        if tag not in self:
            return best

        start_index: int = 0
        rules_for_tag: Dict[Suffix, List[SuffixWithEndMarker]] = self[tag]
        while start_index <= len(token):
            token_suffix: Suffix = token[start_index:]
            if token_suffix in rules_for_tag:
                best = TokenTransformations(token_suffix, rules_for_tag[token_suffix])
                break
            start_index += 1
        return best

    def has_rule_for(self, tag: Tag, token_suffix: Suffix, lemma_suffix: Suffix):
        lemma_suffixes: List[SuffixWithEndMarker] = self[tag][token_suffix]
        return lemma_suffix in (ls.suffix for ls in lemma_suffixes)

    def has_tag(self, tag: Tag):
        return tag in self

    def is_rule_locked_for(self, tag: Tag, full_form_suffix: Suffix):
        rules_list: List[SuffixWithEndMarker] = self[tag][full_form_suffix]
        if not rules_list:
            return False
        return rules_list[0].end_marker

    # def __len__(self) -> int:
    #     return sum(len(lemmas) for suffix_lookup in self.values() for lemmas in suffix_lookup.values())


T = TypeVar("T")


class Counter(Generic[T], Serializable["Counter[T]"]):
    @classmethod
    def _version(cls) -> int:
        return 1

    def _to_bytes(self) -> Dict[str, Any]:
        return dict(
            total=self._total,
            counts=self._counts
        )

    @classmethod
    def _from_bytes(cls, msg: Dict[str, Any]) -> "Counter[T]":
        counter = Counter()
        counter._counts = defaultdict(int, msg["counts"])
        counter._total = msg["total"]
        return counter

    def __init__(self):
        self._counts: Dict[T: int] = defaultdict(int)
        self._total: int = 0

    def add(self, item: T):
        self._counts[item] += 1
        self._total += 1

    def add_all(self, items: Iterable[T]):
        for e in items:
            self.add(e)

    def __getitem__(self, item: T) -> int:
        return self._counts[item]

    def __len__(self):
        return self._total


# noinspection PyPep8Naming
class Lemmatizer(Serializable["Lemmatizer"]):  # pylint: disable=too-few-public-methods
    """Class for lemmatizing words. Inspired by the CST lemmatizer."""

    _logger = logging.getLogger(__name__)

    @classmethod
    def _version(cls) -> int:
        return 1

    def _to_bytes(self) -> Dict[str, Any]:
        msg = dict(
            rule_repo=self._rule_repo.to_bytes(),
            lemma_counter=self._lemma_counter.to_bytes()
        )
        return msg

    @classmethod
    def _from_bytes(cls, msg: Dict[str, Any]) -> "Lemmatizer":
        lemmatizer = Lemmatizer()
        lemmatizer._rule_repo = RuleRepository.from_bytes(msg["rule_repo"])
        lemmatizer._lemma_counter = Counter[Lemma].from_bytes(msg["lemma_counter"])
        return lemmatizer

    def __init__(self):
        """Initialize a lemmatizer using specified set of rules."""
        self._rule_repo = RuleRepository()
        self._lemma_counter: Counter[Lemma] = Counter[Lemma]()

    def fit(self, tags_with_tokens: Iterable[Tuple[Tag, Token, Position]], lemmata: Iterable[Lemma], max_iteration: int = 20):
        """Train a lemmatizer on specified training data."""
        self._lemma_counter.add_all(lemmata)

        old_rule_count = -1
        epoch = 1
        train_start: float = time.time()
        rule_count: int = 0
        while old_rule_count != len(self._rule_repo) and epoch <= max_iteration:
            epoch_start: float = time.time()
            old_rule_count: int = len(self._rule_repo)
            self._train_epoch(tags_with_tokens, lemmata)
            rule_count = len(self._rule_repo)
            self._logger.debug("epoch #%s: %s rules (%s new) in %.2fs", epoch, rule_count, rule_count - old_rule_count,
                               time.time() - epoch_start)
            epoch += 1
        self._logger.debug("training complete: %s rules in %.2fs", rule_count, time.time() - train_start)
        self._prune(tags_with_tokens)

    def _train_epoch(self, tags_with_tokens: Iterable[Tuple[Tag, Token, Position]], lemmata: Iterable[Lemma]):
        tag: Tag
        token: Token
        lemma: Lemma
        for (tag, token, position), lemma in zip(tags_with_tokens, lemmata):
            token = self.__mask_numbers(token)
            lemma = self.__mask_numbers(lemma)
            rule: TokenTransformations = self._rule_repo.get_longest_matching_rule_for(tag, token)
            predicted_lemmas: List[Lemma] = rule(token)

            if len(predicted_lemmas) == 1 and lemma in predicted_lemmas:
                # Current rules yield the correct lemma, so nothing to do.
                continue

            # Current rules don't yield the correct lemma, so we will add a new rule. To make sure the new rule will
            # be used, try to make it at least one character longer than existing longest matching rule.
            current_rule_length: int = len(rule.token_suffix)

            rule: TokenTransformations = TokenTransformations.create(token, lemma, current_rule_length)
            lemma_suffixes: List[SuffixWithEndMarker] = rule.lemma_suffixes
            exhausted: bool = lemma_suffixes[0].end_marker

            if exhausted:
                # New rule exhausts full form, meaning we may or may not have an existing rule with the new full
                # form suffix.
                if not self._rule_repo.is_rule_locked_for(tag, rule.token_suffix):
                    # Existing rules for the full form suffix (if there are any) are not locked. So we remove them
                    # and replace them with the new rule.
                    self._rule_repo[tag][rule.token_suffix] = lemma_suffixes
                elif not self._rule_repo.has_rule_for(tag, rule.token_suffix, lemma_suffixes[0].suffix):
                    self._rule_repo[tag][rule.token_suffix] += lemma_suffixes
            else:
                # New rule does not exhaust full form, meaning we are not using the complete full form for suffix.
                # Therefore it's safe to assume the new rule is longer than previous matching rule, and subsequently
                # that no rules with the new suffix exist. And so, we don't have to consider existing rules.
                self._rule_repo[tag][rule.token_suffix] = lemma_suffixes

    def _prune(self, tags_with_tokens: Iterable[Tuple[Tag, Token, Position]]):
        pre_prune_count: int = len(self._rule_repo)
        self._logger.debug("rules before pruning: %s", pre_prune_count)
        used_rules: Dict = {}

        tag: Tag
        token: Token
        for tag, token, position in tags_with_tokens:
            rule: TokenTransformations = self._rule_repo.get_longest_matching_rule_for(tag, token)
            if rule.token_suffix == "" and rule.lemma_suffixes[0].suffix == "":
                continue
            used_rules[tag + "_" + rule.token_suffix] = len(rule.lemma_suffixes)

        self._logger.debug("used rules: %s", sum(used_rules.values()))

        tag: Tag
        for tag, lemma_suffixes_for_token_suffix in self._rule_repo.items():
            token_suffixes = list(lemma_suffixes_for_token_suffix.keys())
            for token_suffix in token_suffixes:
                if tag + "_" + token_suffix not in used_rules:
                    lemma_suffixes_for_token_suffix.pop(token_suffix)

        post_prune_count = len(self._rule_repo)
        self._logger.debug("rules after pruning: %s (%s removed)", post_prune_count, pre_prune_count - post_prune_count)

    def lemmatize(self, tag: Tag, token: Token, position: Position, prev_tag: Optional[Tag] = None,
                  disambiguate: bool = True) -> Union[Lemma, List[Lemma]]:
        """Return lemma for specified full form word of specified word class."""
        token = self.__simple_true_casing(tag, token, position)
        masked_token = self.__mask_numbers(token)
        rule: TokenTransformations = self._rule_repo.get_longest_matching_rule_for(tag, masked_token)
        predicted_lemmas: List[Lemma] = rule(masked_token)

        lemma_candidates: List[Lemma] = predicted_lemmas #+ list(map(str.lower, predicted_lemmas))

        # Lemmatize using history. FIXME: most probably this never happens
        if prev_tag is not None and self._rule_repo.has_tag(prev_tag + "_" + tag):
            lemma_candidates = self.lemmatize(prev_tag + "_" + tag, masked_token, prev_tag=None)

        if not disambiguate:
            return lemma_candidates

        new_lemma_candidates = list()
        for lemma_candidate in lemma_candidates:
            if masked_token != token:
                if lemma_candidate[-1:] == '-':
                    lemma_candidate = lemma_candidate[:-1]
                new_lemma_candidates.append(token[:len(lemma_candidate)])
            else:
                new_lemma_candidates.append(lemma_candidate)

        lemma: Lemma = self.disambiguate(tag, token, new_lemma_candidates)

        if self._lemma_counter[lemma] <= 0 and self._lemma_counter[token] > 0:
            lemma = token

        return lemma

    def disambiguate(self, tag: Tag, token: Token, lemma_candidates: List[Lemma]) -> Lemma:
        max_count = -1
        lemma: Lemma = token
        for candidate in lemma_candidates:
            if len(candidate) > len(token):
                continue
            count: int = self._lemma_counter[candidate]
            if count > max_count:
                lemma = candidate
                max_count = count

        # noinspection PyUnboundLocalVariable
        return lemma

    @staticmethod
    def __simple_true_casing(tag: Tag, token: Token, position: Position) -> Token:
        if position != 0:
            return token
        if token.islower():
            return token
        if tag != 'PROPN':
            token = token.lower()
        return token

    @staticmethod
    def __mask_numbers(token: Token) -> Token:
        return re.sub(r'\d', '0', token)


def _find_suffix_start(token: Token, lemma: Lemma, min_rule_length: int) -> int:
    """
    Find and return the index at which the suffix begins.

    Full form and lemma will have all characters to the left of the suffix in common. The split will be made far enough
    to the left to allow for the suffix to consist of at least 'min_rule_length' characters.
    """

    token_prefix_len: int = len(token) - min_rule_length
    max_prefix_length: int = min(len(lemma), token_prefix_len)
    suffix_start: int = 0
    while suffix_start < max_prefix_length and lemma[suffix_start] == token[suffix_start]:
        suffix_start += 1
    return suffix_start
