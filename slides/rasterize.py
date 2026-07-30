"""Rasterize the ACM report's vector figures to PNG for slide embedding."""
import os

import fitz  # pymupdf

SRC = os.path.join(os.path.dirname(__file__), "..", "report", "acm", "figures")
DST = os.path.join(os.path.dirname(__file__), "assets")
DPI = 220

os.makedirs(DST, exist_ok=True)
for name in ["fig_baseline", "fig_signals", "fig_ablation", "fig_tradeoff"]:
    doc = fitz.open(os.path.join(SRC, f"{name}.pdf"))
    page = doc[0]
    pix = page.get_pixmap(dpi=DPI)
    pix.save(os.path.join(DST, f"{name}.png"))
    doc.close()
    print(f"{name}.png written")
