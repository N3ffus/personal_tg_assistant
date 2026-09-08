"""Conservative title matching for the watched catalogue."""

import re
import unicodedata


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
