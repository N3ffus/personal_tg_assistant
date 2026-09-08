"""Conservative title matching for the watched catalogue."""

import re
import unicodedata
from collections.abc import Iterable
from typing import Protocol


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
