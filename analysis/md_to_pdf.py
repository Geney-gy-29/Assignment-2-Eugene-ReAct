"""One-off: convert report/Baseline_Reproduction_Report.md to PDF."""
import markdown
from xhtml2pdf import pisa

SRC = "report/Baseline_Reproduction_Report.md"
DST = "report/Baseline_Reproduction_Report.pdf"

with open(SRC, encoding="utf-8") as f:
    body = markdown.markdown(f.read(), extensions=["tables"])

html = f"""<html><head><style>
body {{ font-family: Helvetica, sans-serif; font-size: 10.5pt; }}
h1 {{ font-size: 16pt; }}
h2 {{ font-size: 13pt; margin-top: 16pt; }}
h3 {{ font-size: 11.5pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 8pt 0; }}
th, td {{ border: 1px solid #999; padding: 4pt 6pt; font-size: 9.5pt; }}
th {{ background: #eee; }}
code {{ background: #f2f2f2; }}
</style></head><body>{body}</body></html>"""

with open(DST, "wb") as f:
    pisa.CreatePDF(html, dest=f)
print(f"Wrote {DST}")
