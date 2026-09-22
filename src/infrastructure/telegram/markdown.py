"""Markdown written by the LLM, rendered as the HTML subset Telegram accepts.

Telegram shows `**bold**` literally, and its own MarkdownV2 rejects any
unescaped `.`/`-`/`!`, so a model reply can never be sent as Markdown directly.
Every tag produced here is closed on the same line or block it was opened in,
which keeps the result well-formed for any input.
"""

import re
from html import escape

FENCE = re.compile(r"^\s*```")
HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
BULLET = re.compile(r"^(\s*)[-*+]\s+(.*)$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")

CODE_SPAN = re.compile(r"`([^`\n]+)`")
LINK = re.compile(r"\[([^\]\n]+)\]\(((?:https?://|tg://)[^\s)]+)\)")
BOLD_ITALIC = re.compile(r"\*\*\*(?=\S)([^*\n]+?)(?<=\S)\*\*\*")
# No `__bold__`: models rarely write it, and `__init__` must stay a name.
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
# Italic and strikethrough never span a tag: the result can not cross-nest.
ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=[^\s*])([^*\n<>]+?)(?<=[^\s*])\*(?![\w*])")
ITALIC_UNDERSCORE = re.compile(
    r"(?<![\w_])_(?=[^\s_])([^_\n<>]+?)(?<=[^\s_])_(?![\w_])"
)
STRIKE = re.compile(r"~~(?=\S)([^~\n<>]+?)(?<=\S)~~")
PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def markdown_to_html(text: str) -> str:
    out: list[str] = []
    quote: list[str] = []
    code: list[str] | None = None

    def flush_quote() -> None:
        if quote:
            out.append(f"<blockquote>{chr(10).join(quote)}</blockquote>")
            quote.clear()

    for line in text.split("\n"):
        if FENCE.match(line):
            if code is None:
                flush_quote()
                code = []
            else:
                out.append(f"<pre>{escape(chr(10).join(code))}</pre>")
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if quoted := QUOTE.match(line):
            quote.append(_inline(quoted[1]))
            continue
        flush_quote()
        if heading := HEADING.match(line):
            out.append(f"<b>{_inline(heading[1])}</b>")
        elif RULE.match(line):
            out.append("──────────")
        elif bullet := BULLET.match(line):
            out.append(f"{bullet[1]}• {_inline(bullet[2])}")
        else:
            out.append(_inline(line))
    flush_quote()
    if code is not None:
        out.append(f"<pre>{escape(chr(10).join(code))}</pre>")
    return "\n".join(out)


def _inline(text: str) -> str:
    kept: list[str] = []

    def keep(html: str) -> str:
        kept.append(html)
        return f"\x00{len(kept) - 1}\x00"

    text = text.replace("\x00", "")
    text = CODE_SPAN.sub(lambda m: keep(f"<code>{escape(m[1])}</code>"), text)
    text = LINK.sub(
        lambda m: keep(f'<a href="{escape(m[2])}">{escape(m[1])}</a>'), text
    )
    text = escape(text, quote=False)
    text = BOLD_ITALIC.sub(r"<b><i>\1</i></b>", text)
    text = BOLD.sub(r"<b>\1</b>", text)
    text = ITALIC_STAR.sub(r"<i>\1</i>", text)
    text = ITALIC_UNDERSCORE.sub(r"<i>\1</i>", text)
    text = STRIKE.sub(r"<s>\1</s>", text)
    return PLACEHOLDER.sub(lambda m: kept[int(m[1])], text)
