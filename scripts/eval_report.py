"""Render a DeepEval run report from ``eval-results/junit.xml`` as a PDF.

Run after ``make eval``::

    make eval-report

The report keeps the absolute metric scores the judge produced: a pass/fail
summary hides how close a scenario ran to its threshold.
"""

# The only input is the JUnit file this repository's own pytest run just
# wrote, so the stdlib parser needs no hardening against hostile XML.
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

JUNIT = Path("eval-results/junit.xml")
OUTPUT = Path("eval-results/deepeval-report.pdf")
# The judge scores quality on a 0..1 scale; tool selection must match exactly.
QUALITY_THRESHOLD = 0.8
TOOLS_THRESHOLD = 1.0

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#d8d8d8")
BAND = colors.HexColor("#f5f5f3")
PASS = colors.HexColor("#1f7a3f")
FAIL = colors.HexColor("#b3261e")
WARN = colors.HexColor("#a8690a")


@dataclass(frozen=True)
class Case:
    name: str
    tools: float | None
    quality: float | None
    failed: bool
    reason: str
    seconds: float

    @property
    def scored(self) -> bool:
        return self.quality is not None or self.tools is not None


def register_fonts() -> tuple[str, str]:
    """Use a Cyrillic-capable system font; reportlab's built-ins are Latin-1."""
    candidates = [
        ("DejaVuSans", "DejaVuSans-Bold", "DejaVuSans.ttf", "DejaVuSans-Bold.ttf"),
        ("Arial", "Arial-Bold", "arial.ttf", "arialbd.ttf"),
    ]
    roots = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/dejavu")]
    for regular, bold, regular_file, bold_file in candidates:
        for root in roots:
            if (root / regular_file).exists() and (root / bold_file).exists():
                pdfmetrics.registerFont(TTFont(regular, str(root / regular_file)))
                pdfmetrics.registerFont(TTFont(bold, str(root / bold_file)))
                return regular, bold
    return "Helvetica", "Helvetica-Bold"


def number(value: str | None) -> float | None:
    return float(value) if value else None


def scenario_id(name: str) -> str:
    return name.split("[", 1)[1].rstrip("]") if "[" in name else name


def load_cases(path: Path) -> tuple[ET.Element, list[Case]]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    cases = []
    for case in suite.findall("testcase"):
        props = {
            property.get("name"): property.get("value")
            for property in case.findall("properties/property")
        }
        cases.append(
            Case(
                name=scenario_id(str(case.get("name", ""))),
                tools=number(props.get("tools_score")),
                quality=number(props.get("quality_score")),
                failed=bool(case.findall("failure")),
                reason=(props.get("quality_reason") or "").strip(),
                seconds=float(case.get("time") or 0),
            )
        )
    return suite, cases


def header(text: str, bold: str) -> Paragraph:
    return Paragraph(
        text, ParagraphStyle("th", fontName=bold, fontSize=9, textColor=MUTED)
    )


def verdict(case: Case, bold: str) -> Paragraph:
    colour = FAIL if case.failed else PASS
    return Paragraph(
        f'<font color="#{colour.hexval()[2:]}">'
        f"{'провал' if case.failed else 'ок'}</font>",
        ParagraphStyle("verdict", fontName=bold, fontSize=9),
    )


def score(value: float | None, threshold: float, bold: str) -> Paragraph:
    if value is None:
        return Paragraph('<font color="#6b6b6b">—</font>', ParagraphStyle("na"))
    colour = (
        PASS if value >= threshold else (WARN if value >= threshold - 0.2 else FAIL)
    )
    return Paragraph(
        f"{value:.2f}",
        ParagraphStyle("score", fontName=bold, fontSize=9, textColor=colour),
    )


def banded(rows: list[list[Flowable]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    # Table commands are heterogeneous tuples; the stubs describe them per shape.
    style: list[Any] = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.25, RULE),
        ("LINEBELOW", (0, -1), (-1, -1), 0.7, RULE),
    ]
    style += [
        ("BACKGROUND", (0, index), (-1, index), BAND)
        for index in range(2, len(rows), 2)
    ]
    table.setStyle(TableStyle(style))
    return table


def build(path: Path = JUNIT, output: Path = OUTPUT) -> Path:
    regular, bold = register_fonts()
    suite, cases = load_cases(path)
    scored = [case for case in cases if case.scored]
    unscored = [case for case in cases if not case.scored]
    failed = [case for case in cases if case.failed]
    qualities = [case.quality for case in scored if case.quality is not None]
    tools = [case.tools for case in scored if case.tools is not None]

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "body",
        parent=styles["Normal"],
        fontName=regular,
        fontSize=9.5,
        leading=13.5,
        textColor=INK,
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=8.5, leading=12, textColor=MUTED
    )
    cell = ParagraphStyle("cell", parent=body, fontSize=9, leading=11.5)
    title = ParagraphStyle(
        "title", parent=body, fontName=bold, fontSize=20, leading=24, spaceAfter=2
    )
    section = ParagraphStyle(
        "section",
        parent=body,
        fontName=bold,
        fontSize=12,
        leading=16,
        spaceBefore=14,
        spaceAfter=6,
    )

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="DeepEval — отчёт о прогоне",
        author="personal-ai-assistant",
    )
    timestamp = str(suite.get("timestamp", ""))[:19].replace("T", " ")
    story: list[Flowable] = [
        Paragraph("DeepEval — отчёт о прогоне", title),
        Paragraph(
            f"personal-ai-assistant · прогон {timestamp} · "
            f"длительность {float(suite.get('time') or 0) / 60:.1f} мин",
            small,
        ),
        Spacer(1, 10),
    ]

    summary = [
        ("Всего сценариев", str(len(cases))),
        ("Со скорингом", str(len(scored))),
        ("Пройдено", str(len(cases) - len(failed))),
        ("Упало", str(len(failed))),
        (
            "Средний quality",
            f"{sum(qualities) / len(qualities):.3f}" if qualities else "—",
        ),
        ("Средний tools", f"{sum(tools) / len(tools):.3f}" if tools else "—"),
    ]
    totals = Table(
        [
            [Paragraph(label, small) for label, _ in summary],
            [
                Paragraph(
                    value,
                    ParagraphStyle("total", fontName=bold, fontSize=15, textColor=INK),
                )
                for _, value in summary
            ],
        ],
        colWidths=[29 * mm] * len(summary),
    )
    totals.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LINEABOVE", (0, 0), (-1, 0), 0.7, RULE),
                ("LINEBELOW", (0, -1), (-1, -1), 0.7, RULE),
            ]
        )
    )
    story += [
        totals,
        Spacer(1, 6),
        Paragraph(
            f"<b>tools</b> — ToolCorrectness, порог {TOOLS_THRESHOLD:.1f}: точное "
            f"совпадение набора и порядка вызовов интеграций. <b>quality</b> — "
            f"GEval-судья (zai-org/GLM-5.3-Flash), порог {QUALITY_THRESHOLD:.1f}. "
            f"Баллы абсолютные, шкала 0…1.",
            small,
        ),
        Paragraph("Сценарии промпта и интеграций", section),
    ]

    rows: list[list[Flowable]] = [
        [header(name, bold) for name in ("Сценарий", "tools", "quality", "сек", "итог")]
    ]
    rows += [
        [
            Paragraph(case.name.replace("_", " "), cell),
            score(case.tools, TOOLS_THRESHOLD, bold),
            score(case.quality, QUALITY_THRESHOLD, bold),
            Paragraph(f"{case.seconds:.1f}", small),
            verdict(case, bold),
        ]
        for case in scored
    ]
    story.append(banded(rows, [96 * mm, 16 * mm, 18 * mm, 14 * mm, 20 * mm]))

    if failed:
        story.append(Paragraph("Что упало", section))
        for case in failed:
            story.append(Paragraph(f"<b>{case.name.replace('_', ' ')}</b>", body))
            if case.quality is not None:
                story.append(
                    Paragraph(
                        f"quality {case.quality:.2f} при пороге {QUALITY_THRESHOLD:.1f}",
                        small,
                    )
                )
            if case.reason:
                story.append(Paragraph(case.reason, small))
            story.append(Spacer(1, 6))

    if unscored:
        story += [
            PageBreak(),
            Paragraph("Бизнес-диалоги: владение задачей", section),
            Paragraph(
                "Эти сценарии проверяются жёсткими утверждениями, без судьи: "
                "у них нет балла, только вердикт.",
                small,
            ),
            Spacer(1, 6),
        ]
        rows = [[header(name, bold) for name in ("Сценарий", "сек", "итог")]]
        rows += [
            [
                Paragraph(case.name.replace("_", " "), cell),
                Paragraph(f"{case.seconds:.1f}", small),
                verdict(case, bold),
            ]
            for case in unscored
        ]
        story.append(banded(rows, [126 * mm, 14 * mm, 24 * mm]))

    story += [
        Spacer(1, 12),
        Paragraph(
            f"Сгенерировано {datetime.now():%d.%m.%Y %H:%M} из {path.as_posix()}", small
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    document.build(story)
    return output


def main() -> int:
    if not JUNIT.exists():
        print(f"{JUNIT} not found — run `make eval` first")
        return 1
    print(f"-> {build()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
