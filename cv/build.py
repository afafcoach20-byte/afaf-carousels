"""Build the CV: fill icons/photo into the template, then render PDF + PNG.

Usage: python3 build.py [photo_path]
"""
import base64, mimetypes, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
ICONS = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "icons"
HANDLE = "@afafcoach20"

html = (HERE / "cv-template.html").read_text()
for name in ["youtube", "tiktok", "instagram", "linkedin", "telegram"]:
    html = html.replace("{{%s}}" % name, (ICONS / f"{name}.d").read_text().strip())
html = html.replace("{{HANDLE}}", HANDLE)

photo = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else None
if photo:
    mime = mimetypes.guess_type(photo)[0] or "image/jpeg"
    data = base64.b64encode(pathlib.Path(photo).read_bytes()).decode()
    html = html.replace("{{PHOTO}}", f'<img src="data:{mime};base64,{data}" alt="">')
else:
    html = html.replace("{{PHOTO}}", '<svg viewBox="0 0 100 100" style="width:80%;color:#c8957a;opacity:.55"><circle cx="50" cy="38" r="18" fill="currentColor"/><path d="M14 100c4-24 18-36 36-36s32 12 36 36z" fill="currentColor"/></svg>')

(HERE / "afaf-cv.html").write_text(html)
subprocess.run(["node", "-e", """
const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();
const p=await b.newPage({viewport:{width:794,height:1123},deviceScaleFactor:2});
await p.goto('file://'+process.argv[1],{waitUntil:'networkidle'});
await p.pdf({path:process.argv[2],format:'A4',printBackground:true});
await p.screenshot({path:process.argv[3]});await b.close()})()""",
    str(HERE / "afaf-cv.html"), str(HERE / "afaf-cv.pdf"), str(HERE / "afaf-cv.png")], check=True)
