"""
PDF Report Generation Utility
=============================
Converts GitHub-flavored Markdown evaluation reports into publication-quality
PDF documents using Playwright Chromium headless rendering.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger("CWT-PDFGenerator")


def convert_markdown_to_pdf(
    markdown_path: Union[str, Path],
    pdf_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """
    Convert a Markdown file into a styled PDF document using Playwright Chromium.

    Args:
        markdown_path: Path to source Markdown file.
        pdf_path: Destination path for PDF. Defaults to same name with .pdf extension.

    Returns:
        Path to the generated PDF file if successful, None otherwise.
    """
    md_file = Path(markdown_path)
    if not md_file.exists():
        logger.error("Markdown file not found: %s", md_file)
        return None

    out_pdf = Path(pdf_path) if pdf_path else md_file.with_suffix(".pdf")
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    try:
        import markdown_it
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        logger.warning(
            "PDF generation prerequisites missing (%s). Ensure markdown-it-py and playwright are installed.", e
        )
        return None

    try:
        md_text = md_file.read_text(encoding="utf-8")
        md_parser = markdown_it.MarkdownIt().enable("table")
        html_body = md_parser.render(md_text)

        styled_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{md_file.stem.replace('_', ' ').title()}</title>
<style>
@page {{
    size: A4;
    margin: 15mm 12mm 15mm 12mm;
    @bottom-right {{
        content: "Page " counter(page);
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 8pt;
        color: #57606a;
    }}
}}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    color: #1f2328;
    line-height: 1.45;
    font-size: 10pt;
    margin: 0;
    padding: 0;
}}
h1 {{
    color: #0969da;
    border-bottom: 2px solid #0969da;
    padding-bottom: 6px;
    margin-top: 10px;
    font-size: 18pt;
}}
h2 {{
    color: #1f2328;
    border-bottom: 1px solid #d8dee4;
    padding-bottom: 4px;
    margin-top: 16px;
    font-size: 13pt;
}}
h3 {{
    color: #24292f;
    margin-top: 12px;
    font-size: 11pt;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}}
th, td {{
    border: 1px solid #d0d7de;
    padding: 4px 7px;
    text-align: left;
}}
th {{
    background-color: #f6f8fa;
    font-weight: 600;
}}
tr:nth-child(even) {{
    background-color: #fbfbfb;
}}
code {{
    background-color: rgba(175, 184, 193, 0.2);
    padding: 1px 4px;
    border-radius: 3px;
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 8.5pt;
}}
pre code {{
    display: block;
    padding: 8px;
    background-color: #f6f8fa;
    border: 1px solid #d0d7de;
    border-radius: 4px;
    overflow-x: auto;
}}
blockquote {{
    border-left: 4px solid #0969da;
    padding: 4px 10px;
    color: #57606a;
    background-color: #f6f8fa;
    margin: 8px 0;
}}
hr {{
    border: 0;
    height: 1px;
    background: #d8dee4;
    margin: 15px 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(styled_html)
            page.pdf(
                path=str(out_pdf),
                format="A4",
                margin={"top": "15mm", "bottom": "15mm", "left": "12mm", "right": "12mm"},
                print_background=True,
            )
            browser.close()

        logger.info("Successfully compiled PDF report: %s (%d bytes)", out_pdf, out_pdf.stat().st_size)
        return out_pdf

    except Exception as e:
        logger.error("Failed to compile PDF report: %s", e)
        return None
