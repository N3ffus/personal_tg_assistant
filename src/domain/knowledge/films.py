"""Conservative title matching for the watched catalogue."""

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class FilmRecord:
    """A film an external directory confirms exists."""

    title: str
    original_title: str
    year: int | None


def title_key(title: str) -> str:
    text = unicodedata.normalize("NFKC", title).casefold().replace("ё", "е")
    text = re.sub(r"\s*\((?:19|20)\d{2}\)\s*$", "", text)
    return "".join(char for char in text if char.isalnum())


def title_keys(title: str) -> set[str]:
    keys = {title_key(title)}
    # Catalogue titles may include a franchise prefix omitted by a candidate.
    # Exclude ambiguous franchise matches conservatively, but not short words.
    keys.update(key for part in title.split(":") if len(key := title_key(part)) >= 8)
    return keys - {""}


MAX_RECOMMENDATIONS = 10

# Whole word forms: a stem like «сем» would read «семейный фильм» as seven.
_COUNT_WORDS = {
    **dict.fromkeys(["один", "одну", "одного"], 1),
    **dict.fromkeys(["пару", "парочку", "два", "две", "двух"], 2),
    **dict.fromkeys(["три", "трех"], 3),
    **dict.fromkeys(["четыре", "четырех"], 4),
    **dict.fromkeys(["пять", "пяти"], 5),
    **dict.fromkeys(["шесть", "шести"], 6),
    **dict.fromkeys(["семь", "семи"], 7),
    **dict.fromkeys(["восемь", "восьми"], 8),
    **dict.fromkeys(["девять", "девяти"], 9),
    **dict.fromkeys(["десять", "десяти", "десяток"], 10),
    **dict.fromkeys(["дюжину"], 12),
}
_COUNT = r"(\d{1,2}|" + "|".join(_COUNT_WORDS) + r")(?![\w-])"
# «топ 10», «топ-5» or a count before the noun with up to two words between
# them: «7 лучших фильмов», «десять хороших фильмов на вечер».
_REQUESTED_COUNT = re.compile(
    r"\b(?:топ|top)[\s-]*"
    + _COUNT
    + r"|(?<![\w-])"
    + _COUNT
    + r"\s+(?:[а-я]+\s+){0,2}(?:фильм|кино|картин)",
    re.IGNORECASE,
)


def requested_count(text: str) -> int | None:
    """The number of films the user explicitly asked for, if any.

    The model dropped «топ 10» and kept its default of three (production,
    2026-09-15), so the application reads the count itself. Anything above
    `MAX_RECOMMENDATIONS` is capped rather than refused.
    """
    match = _REQUESTED_COUNT.search(text.replace("ё", "е").replace("Ё", "Е"))
    if match is None:
        return None
    token = (match.group(1) or match.group(2)).casefold()
    count = int(token) if token.isdigit() else _COUNT_WORDS[token]
    return min(count, MAX_RECOMMENDATIONS) if count > 0 else None


class Candidate(Protocol):
    """The part of a recommendation candidate that identifies a film."""

    title: str
    aliases: list[str]


def select_unwatched[T: Candidate](
    candidates: Iterable[T], *, watched: set[str], limit: int
) -> list[T]:
    """Keep candidates the catalogue does not already hold, at most `limit`.

    A candidate matched by any alias is watched; a title repeated under another
    spelling is kept only once. The caller owns the catalogue: an empty
    `watched` means nothing was verified, so nothing survives.
    """
    selected: list[T] = []
    if not watched:
        return selected
    excluded = set(watched)
    for candidate in candidates:
        keys = {
            key
            for title in [candidate.title, *candidate.aliases]
            for key in title_keys(title)
        }
        if not title_key(candidate.title) or keys & excluded:
            continue
        excluded.update(keys)
        selected.append(candidate)
        if len(selected) == limit:
            break
    return selected
