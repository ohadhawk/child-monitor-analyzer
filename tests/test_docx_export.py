"""
Regression tests for the exported .docx.

Hebrew transcripts are uploaded to Google Drive, where Drive converts the
.docx into a native Google Doc. That converter is stricter than Word, so two
mistakes that Word silently tolerates both produced a visibly wrong document:

* ``w:bidi`` appended after ``w:jc`` violates the CT_PPr element order, and the
  converter drops the misplaced element -- losing right-to-left direction.
* ``w:jc`` is *logical*, not physical: ``right`` means the end edge, which in a
  right-to-left paragraph is the **left** one. Setting it visibly left-aligned
  the whole transcript.

Both were confirmed against a real Google Docs render before being fixed here,
so these tests pin the exact XML rather than the intent.
"""

from __future__ import annotations

import io
import re
import zipfile

import pytest

pytest.importorskip("docx", reason="python-docx is an optional export dependency")
pytest.importorskip("PySide6", reason="the exporter lives on the GUI widget")

from monitor.gui.transcript_widget import EXPORT_FONT, TranscriptWidget  # noqa: E402

LINES = ["[00:00:01] שלום עולם", "", "שורה שנייה 42 English"]


@pytest.fixture(scope="module")
def document_xml() -> str:
    buffer = io.BytesIO()
    TranscriptWidget._write_docx(buffer, LINES)
    archive = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    return archive.read("word/document.xml").decode("utf-8")


@pytest.fixture(scope="module")
def styles_xml() -> str:
    buffer = io.BytesIO()
    TranscriptWidget._write_docx(buffer, LINES)
    archive = zipfile.ZipFile(io.BytesIO(buffer.getvalue()))
    return archive.read("word/styles.xml").decode("utf-8")


def _body(document_xml: str) -> str:
    return document_xml[document_xml.index("<w:body>"):document_xml.index("<w:sectPr")]


# ===========================
# PARAGRAPH DIRECTION
# ===========================

def test_every_paragraph_is_marked_right_to_left(document_xml):
    body = _body(document_xml)
    assert body.count("<w:p>") == body.count("<w:bidi/>") == len(LINES)


def test_the_normal_style_is_right_to_left(styles_xml):
    """Empty paragraphs inherit direction from the style, not from a run."""
    start = styles_xml.index('w:styleId="Normal"')
    assert "<w:bidi/>" in styles_xml[start:start + 400]


def test_runs_are_marked_right_to_left(document_xml):
    """Without w:rtl, neutrals such as "[00:00:01]" reorder incorrectly."""
    body = _body(document_xml)
    assert body.count("<w:rtl/>") == len(LINES)


# ===========================
# THE TWO CONVERTER TRAPS
# ===========================

def test_bidi_precedes_jc_in_element_order(document_xml):
    """CT_PPr requires w:bidi before w:jc; Google Docs enforces it."""
    body = _body(document_xml)
    for p_pr in body.split("<w:pPr>")[1:]:
        p_pr = p_pr[:p_pr.index("</w:pPr>")]
        if "<w:jc" in p_pr:
            assert p_pr.index("<w:bidi") < p_pr.index("<w:jc")


def test_no_explicit_alignment_is_written(document_xml):
    """w:jc is logical: "right" is the end edge, i.e. left under RTL.

    The w:bidi default already aligns to the start edge, so any w:jc here is
    more likely to flip the transcript than to help it.
    """
    assert "<w:jc" not in _body(document_xml)


def test_the_normal_style_does_not_force_alignment(styles_xml):
    start = styles_xml.index('w:styleId="Normal"')
    assert "<w:jc" not in styles_xml[start:start + 400]


# ===========================
# FONT
# ===========================

def test_the_font_covers_complex_script(styles_xml):
    """Hebrew is a complex script, so it takes w:cs, not w:ascii/w:hAnsi."""
    start = styles_xml.index('w:styleId="Normal"')
    normal = styles_xml[start:start + 400]
    fonts = re.search(r"<w:rFonts([^/]*)/>", normal)
    assert fonts is not None, normal
    for attribute in ("w:ascii", "w:hAnsi", "w:cs"):
        assert f'{attribute}="{EXPORT_FONT}"' in fonts.group(1)
