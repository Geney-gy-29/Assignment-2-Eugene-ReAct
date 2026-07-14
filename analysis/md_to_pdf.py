"""One-off: convert report/Baseline_Reproduction_Report.md to PDF."""
import os
import re

import markdown
from xhtml2pdf import pisa

REPORT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/../report"
REPORT_DIR = os.path.normpath(REPORT_DIR)
SRC = os.path.join(REPORT_DIR, "Baseline_Reproduction_Report.md")
DST = os.path.join(REPORT_DIR, "Baseline_Reproduction_Report.pdf")

with open(SRC, encoding="utf-8") as f:
    body = markdown.markdown(f.read(), extensions=["tables"])

# Resolve relative image paths (e.g. ../analysis/figures/x.png) to absolute
# file:// URIs so xhtml2pdf can embed them regardless of the process cwd.
def _abs_img(m):
    rel = m.group(1)
    abs_path = os.path.normpath(os.path.join(REPORT_DIR, rel))
    return f'src="{abs_path}"'

body = re.sub(r'src="([^"]+)"', _abs_img, body)

html = f"""<html><head><style>
body {{ font-family: Helvetica, sans-serif; font-size: 10.5pt; }}
h1 {{ font-size: 16pt; }}
h2 {{ font-size: 13pt; margin-top: 16pt; }}
h3 {{ font-size: 11.5pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
th, td {{ border: 1px solid #999; padding: 4pt 6pt; font-size: 9.5pt; }}
th {{ background: #eee; }}
code {{ background: #f2f2f2; }}
img {{ max-width: 480pt; }}
</style></head><body>{body}</body></html>"""

with open(DST, "wb") as f:
    pisa.CreatePDF(html, dest=f)
print(f"Wrote {DST}")
