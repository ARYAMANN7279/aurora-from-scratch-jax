"""The final UGP deck: the UGP-I deck and the UGP-II deck merged into one presentation.

usage: python merge_decks.py   (needs python-pptx; writes ../UGP_Final_Presentation.pptx and ../UGP_Final_speaker_notes.md)
Base: ../../UGP/UGP_presentation.pptx (13.33 x 7.5 in). The UGP-II slides (../../UGP-II/UGP2_presentation_draft.pptx,
10 x 5.625 in) are copied in, scaled by 1.33, and given the UGP-I header band and footer, so the two parts read as one
deck. Joining slides (title, contents, part covers, outcome, both parts together, appendix) are edited copies of
existing slides, in the same style.
"""
import copy
import io
import os
import re

from pptx import Presentation
from pptx.oxml.ns import qn

HERE = os.path.dirname(os.path.realpath(__file__))
A_PATH = os.path.join(HERE, "..", "..", "UGP", "UGP_presentation.pptx")
B_PATH = os.path.join(HERE, "..", "..", "UGP-II", "UGP2_presentation_draft.pptx")
OUT = os.path.join(HERE, "..", "UGP_Final_Presentation.pptx")
NOTES_MD = os.path.join(HERE, "..", "UGP_Final_speaker_notes.md")

A = Presentation(A_PATH)
B = Presentation(B_PATH)
SX = A.slide_width / B.slide_width
SY = A.slide_height / B.slide_height
BLANK = next(l for l in A.slide_layouts if l.name == "Blank")
FOOT1 = "Part I  |  Forecasting real airfoil wakes"
FOOT2 = "Part II  |  What pretraining data buys"


# ---- text helpers -----------------------------------------------------------------------------------------------
def _fill(p, text):
    runs = p.findall(qn("a:r"))
    for x in p.findall(qn("a:br")) + p.findall(qn("a:fld")) + runs[1:]:
        p.remove(x)
    runs[0].find(qn("a:t")).text = text


def set_lines(shape, lines):
    """Replace a shape's text, line by line, keeping each paragraph's formatting (extra lines copy the last one)."""
    ps = [p._p for p in shape.text_frame.paragraphs if p._p.findall(qn("a:r"))]
    allp = [p._p for p in shape.text_frame.paragraphs]
    for p in allp:
        if p not in ps:
            p.getparent().remove(p)
    for p in ps[len(lines):]:
        p.getparent().remove(p)
    ps = ps[:len(lines)]
    for p, line in zip(ps, lines):
        _fill(p, line)
    prev = ps[-1]
    for line in lines[len(ps):]:
        q = copy.deepcopy(prev)
        _fill(q, line)
        prev.addnext(q)
        prev = q


def find(slide, pattern):
    return [s for s in slide.shapes if s.has_text_frame and re.search(pattern, s.text_frame.text)]


def remove(shape):
    shape._element.getparent().remove(shape._element)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# ---- copying slides ---------------------------------------------------------------------------------------------
def scale(el):
    for e in el.iter():
        t, a = e.tag, e.attrib
        if t in (qn("a:off"), qn("a:chOff")):
            a["x"], a["y"] = str(int(int(a["x"]) * SX)), str(int(int(a["y"]) * SY))
        elif t in (qn("a:ext"), qn("a:chExt")) and "cx" in a:
            a["cx"], a["cy"] = str(int(int(a["cx"]) * SX)), str(int(int(a["cy"]) * SY))
        elif t == qn("a:gridCol"):
            a["w"] = str(int(int(a["w"]) * SX))
        elif t == qn("a:tr"):
            a["h"] = str(int(int(a["h"]) * SY))
        elif t in (qn("a:rPr"), qn("a:defRPr"), qn("a:endParaRPr")) and "sz" in a:
            a["sz"] = str(int(round(int(a["sz"]) * SX / 50.0)) * 50)
        elif t == qn("a:spcPts"):
            a["val"] = str(int(int(a["val"]) * SX))
        elif t == qn("a:bodyPr"):
            for k in ("lIns", "tIns", "rIns", "bIns"):
                if k in a:
                    a[k] = str(int(int(a[k]) * SX))
        elif t == qn("a:ln") and "w" in a:
            a["w"] = str(int(int(a["w"]) * SX))
        elif t == qn("a:tcPr"):
            for k in ("marL", "marR", "marT", "marB"):
                if k in a:
                    a[k] = str(int(int(a[k]) * SX))


def copy_shapes(src, dst, elements, rescale):
    tree = dst.shapes._spTree
    for el in elements:
        el = copy.deepcopy(el)
        if rescale:
            scale(el)
        for blip in el.iter(qn("a:blip")):
            rid = blip.get(qn("r:embed"))
            if rid:
                _, new = dst.part.get_or_add_image_part(io.BytesIO(src.part.related_part(rid).blob))
                blip.set(qn("r:embed"), new)
        tree.append(el)


def clone(src, rescale=False, keep_bg=True):
    """A new slide at the end of A with src's background, shapes and notes."""
    dst = A.slides.add_slide(BLANK)
    bg = src._element.cSld.bg
    if keep_bg and bg is not None:
        dst._element.cSld.insert(0, copy.deepcopy(bg))
        if rescale:
            scale(dst._element.cSld.bg)
    copy_shapes(src, dst, [s._element for s in src.shapes], rescale)
    if src.has_notes_slide:
        notes(dst, src.notes_slide.notes_text_frame.text)
    return dst


CHROME = A.slides[1]  # "What is Covered": background, header band, accent strip, title, footer
SUB_TMPL = copy.deepcopy(find(A.slides[16], r"^Current Rank")[0]._element)  # templates, taken before any edit
BODY_TMPL = copy.deepcopy(find(A.slides[20], r"Ship the final")[0]._element)


def add_text(dst, tmpl, lines):
    dst.shapes._spTree.append(copy.deepcopy(tmpl))
    shape = dst.shapes[-1]
    set_lines(shape, lines)
    return shape


def b_content(src):
    """A UGP-II content slide in the UGP-I frame: header band with its title and subtitle, footer."""
    title = [s for s in src.shapes if s.has_text_frame and s.text_frame.paragraphs[0].runs
             and s.text_frame.paragraphs[0].runs[0].font.size and s.text_frame.paragraphs[0].runs[0].font.size.pt == 28]
    sub = [s for s in src.shapes if s.has_text_frame and s.top < 914400 * 1.3 and s not in title and s.text_frame.text.strip()
           and s.text_frame.paragraphs[0].runs and s.text_frame.paragraphs[0].runs[0].font.size and s.text_frame.paragraphs[0].runs[0].font.size.pt == 15]
    num = [s for s in src.shapes if s.has_text_frame and re.fullmatch(r"\d+", s.text_frame.text.strip()) and s.top > 914400 * 5.0]
    dst = A.slides.add_slide(BLANK)
    ch = list(CHROME.shapes)
    copy_shapes(CHROME, dst, [ch[0]._element, ch[1]._element, ch[2]._element], False)
    skip = [s._element for s in title + sub + num]
    copy_shapes(src, dst, [e for e in (s._element for s in src.shapes) if not any(e is k for k in skip)], True)
    copy_shapes(CHROME, dst, [ch[3]._element, ch[4]._element, ch[5]._element, ch[6]._element], False)
    set_lines(find(dst, r"^What is Covered$")[0], [title[0].text_frame.text])
    if sub:
        add_text(dst, SUB_TMPL, [sub[0].text_frame.text])
    if src.has_notes_slide:
        notes(dst, src.notes_slide.notes_text_frame.text)
    return dst


# ---- build ------------------------------------------------------------------------------------------------------
a = list(A.slides)  # a[i] is UGP-I slide i + 1
b = list(B.slides)  # b[i] is UGP-II slide i + 1

# title and thank-you (both light-blue covers, from the UGP-I title slide before it is edited)
title = clone(a[0])
ty = clone(a[0])
set_lines(find(title, r"^UGP Presentation$")[0], ["UGP Final Presentation  ·  Parts I and II"])
set_lines(find(title, r"^Forecasting Real Airfoil")[0], ["Neural Networks for Fluid Flow", "Part I: real airfoil wakes   ·   Part II: pretraining"])
set_lines(find(title, r"^NeurIPS 2026")[0], ["Supervisor: Prof. Rajesh Ranjan   ·   November 2026"])
notes(title, "Good morning, sir. My UGP had two parts. In Part I, I forecast a real airfoil wake in the NeurIPS RealPDE competition. "
             "In Part II, I rebuilt Microsoft's Aurora model to answer the question Part I left open: what does pretraining data actually buy?")

# contents (UGP-I's "What is Covered", rewritten for both parts)
cov = a[1]
for old, new in (("The problem statement", "Part I: the problem"), ("The method", "Part I: our approach"), ("The scoring", "Part I: the result"),
                 ("What we tried", "Part II: the question"), ("Where we stand", "Part II: the findings")):
    set_lines(find(cov, "^" + old + "$")[0], [new])
for old, new in (("Forecasting future frames", "Forecasting a real wake"), ("What the model does", "Model, scoring, bands"), ("Five Metrics to score on", "72.7 → 80.1"),
                 ("Including the failures", "What pretraining buys"), ("Against several teams", "Five results, next steps")):
    set_lines(find(cov, "^" + old + "$")[0], [new])
notes(cov, "The plan: first the competition, its problem, our approach and the result. Then the question it left open, and what Part II found.")

# Part I cover (UGP-I's own title slide)
p1 = a[0]
set_lines(find(p1, r"^UGP Presentation$")[0], ["Part I"])
remove(find(p1, r"^Aryamann")[0])
set_lines(find(p1, r"^NeurIPS 2026")[0], ["NeurIPS 2026 RealPDE Competition  ·  Track 1: Sim2Real"])
notes(p1, "Part I: forecasting real airfoil wakes in the NeurIPS 2026 RealPDE competition.")

# "Where We Stand": the 24 September leaderboard, labelled as such
ws = a[16]
set_lines(find(ws, r"^Current Rank")[0], ["Development leaderboard on 24 September 2026: 181 teams"])
set_lines(find(ws, r"We are currently")[0], ["A very tight field: 2.5% separates rank 1 from outside the top 50."])

# UGP-I's last slide becomes the Part I outcome
oc = a[20]
set_lines(find(oc, r"^Where We Are and What Is Next$")[0], ["Part I: Where We Landed"])
set_lines(find(oc, r"^24 September 2026$")[0], ["Final development leaderboard, about 182 teams"])
for old, val, lab in ((r"^80\.081$", "80.081", "our score"), (r"^82\.101$", "72.65", "organisers' baseline"),
                      (r"^81\.765$", "81.901", "top-10 cutoff"), (r"^61$", "82.218", "development leader")):
    v = find(oc, old)[0]
    set_lines(v, [val])
    labs = [s for s in oc.shapes if s.has_text_frame and abs(s.left - v.left) < 50000 and s.top > v.top and s.top - v.top < 914400 * 0.7]
    set_lines(labs[0], [lab])
set_lines(find(oc, r"^Next$")[0], ["What it led to"])
set_lines(find(oc, r"Ship the final")[0], ["•  Finished just outside the top 50: the whole top 50 sat within 1.6 points.",
                                           "•  The forecast, not the bands, was the limit.",
                                           "•  Open question: would pretraining on related data have helped?",
                                           "•  Part II answers it on a flow we control."])
set_lines(find(oc, r"PERFECT")[0], ["Even a mathematically PERFECT uncertainty model scores 81.727, below the top-10 cutoff (81.765 on 24 Sep)."])
notes(oc, "Where Part I landed. Our development score was 80.08, against the organisers' baseline of 72.65. The top-10 cutoff was 81.90 and the leader 82.22. "
          "Even perfect uncertainty bands would not reach the top ten, so the forecast itself was the limit. We finished just outside the top 50 in a very tight field, "
          "and the project left one question open: would pretraining on related data have helped?")

# Part II cover (UGP-II's own title slide)
p2 = clone(b[0], rescale=True)
set_lines(find(p2, r"^UGP-II$")[0], ["Part II"])
for pat in (r"^Aryamann", r"^Supervisor"):
    remove(find(p2, pat)[0])
from pptx.dml.color import RGBColor  # noqa: E402
for sh in p2.shapes:  # the bottom strip in Part I's navy
    if not sh.has_text_frame or not sh.text_frame.text.strip():
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor(0x1F, 0x4E, 0x79)
notes(p2, "Part II answers that question: what does pretraining data really buy? I rebuilt Aurora, a large weather model, from scratch in JAX and tested it on a swirling flow where I control the data.")

# UGP-II content: 3 to 21, without 13 (how the experiments were run)
b_slides = {i: b_content(b[i - 1]) for i in range(3, 22) if i != 13}
kt = b_slides[20]
set_lines(find(kt, r"^A verified toolkit")[0], ["Building the physics in keeps 20-step forecasts 13% steadier"])
notes(kt, kt.notes_slide.notes_text_frame.text.replace("And I now have a verified toolkit to keep building on.",
                                                       "And building the physics into the model keeps 20-step forecasts 13 percent steadier."))

# both parts together
tog = b_content(b[19])
set_lines(find(tog, r"^Key Takeaways$")[0], ["Both Parts Together"])
add_text(tog, SUB_TMPL, ["What UGP-I and UGP-II teach as one project"])
for old, new in (("Pretraining halves next-step forecast error", "Know what you are measured on: hidden score weights (I), next step vs long forecasts (II)"),
                 ("Closeness to the task matters more than amount or variety", "Pretraining helps when it is close to the task and target data are scarce"),
                 ("The biggest payoff comes when task data is scarce", "Far-away simulations did not help on the real wake (I and II)"),
                 ("Training on its own predictions keeps long forecasts steady", "Physics helps only when it is true for the data: hurt on camera data, helped on simulation"),
                 ("A verified toolkit: simulator, Aurora copy and experiment system", "Write the prediction down first: every surprise becomes a result")):
    set_lines(find(tog, "^" + re.escape(old) + "$")[0], [new])
notes(tog, "Putting both parts together. Know what you are measured on. Pretraining helps when it is close to the task and target data is scarce, and far-away simulations "
           "did not help on the real wake. Physics helps only when it is true for the data: it hurt on the camera data in Part I and helped on the simulation in Part II. "
           "And writing predictions down first turns every surprise into a result.")

# appendix, mentioned only
apx = clone(a[1])
for s in list(apx.shapes)[7:]:
    remove(s)
set_lines(find(apx, r"^What is Covered$")[0], ["Appendix"])
add_text(apx, SUB_TMPL, ["The details are in the final report, not on the slides"])
items = ["A.  Part I: the scoring formulas, the uncertainty bands, the component ablation and the directions we closed",
         "B.  Part II: the solver, dataset, model and training details",
         "C.  All eleven Part II experiments and the full prediction record",
         "D.  Deriving velocity and pressure from the spin, and a glossary",
         "",
         "Final report: UGP_Final_Report.pdf",
         "Part I code and record: github.com/ARYAMANN7279/RealPDE-Track1",
         "Part II code and results: github.com/ARYAMANN7279/aurora-from-scratch-jax"]
bshape = add_text(apx, BODY_TMPL, items)
bshape.left, bshape.top, bshape.width, bshape.height = int(914400 * 0.8), int(914400 * 1.7), int(914400 * 11.7), int(914400 * 4.5)
notes(apx, "The details are in the final report's appendices: the scoring formulas, the model and data, all eleven experiments with the prediction record, and the physics derivation. "
           "I am happy to go into any of them.")

# thank you
set_lines(find(ty, r"^UGP Presentation$")[0], ["UGP  ·  Parts I and II"])
set_lines(find(ty, r"^Forecasting Real Airfoil")[0], ["Thank You"])
set_lines(find(ty, r"^Aryamann")[0], ["Part I:   github.com/ARYAMANN7279/RealPDE-Track1"])
set_lines(find(ty, r"^NeurIPS 2026")[0], ["Part II:  github.com/ARYAMANN7279/aurora-from-scratch-jax"])
notes(ty, "Thank you, sir. I am happy to take questions.")

# ---- order, numbering, footers ---------------------------------------------------------------------------------
part1 = [a[i] for i in list(range(2, 16)) + [16, 17, 18, 20]]  # UGP-I 3-16, 17, 18, 19, 21 (20 dropped)
order = [title, cov, p1] + part1 + [p2] + [b_slides[i] for i in sorted(b_slides)] + [tog, apx, ty]
lst = A.slides._sldIdLst
by_id = {int(el.get("id")): el for el in lst}
sid = {id(s): s.slide_id for s in A.slides}  # before any relationship is dropped
dropped = by_id[a[19].slide_id]  # UGP-I slide 20, "What We Tested and Closed"
for el in list(lst):
    lst.remove(el)
for s in order:
    lst.append(by_id[sid[id(s)]])
A.part.drop_rel(dropped.get(qn("r:id")))
N = len(order)
for k, s in enumerate(order, 1):
    for sh in find(s, r"^\d+ / \d+$"):
        set_lines(sh, [f"{k} / {N}"])
    for sh in find(s, r"^UGP  \|  NeurIPS"):
        set_lines(sh, [FOOT2 if k > order.index(p2) + 1 else FOOT1])
# public-safe: picture alt text without local file paths; plain words for the leaderboard tool
for s in order:
    for el in s._element.iter(qn("p:cNvPr")):
        d = el.get("descr")
        if d and "/" in d:
            el.set("descr", os.path.basename(d))
    for sh in find(s, r"MCP"):
        for para in sh.text_frame.paragraphs:
            if "MCP" in para.text:
                _fill(para._p, "Built a small tool that reads the public Codabench leaderboard, which let us solve for the weights of all metrics"
                      if para.text.startswith("Built") else "A small tool reading Codabench's public leaderboard endpoint")
A.save(OUT)

md = "# UGP Parts I and II: speaker notes\n\nOne deck: Part I (UGP-I) and Part II (UGP-II). About 30 seconds a slide.\n"
for k, s in enumerate(order, 1):
    head = next((sh.text_frame.text for sh in s.shapes if sh.has_text_frame and sh.text_frame.text.strip()
                 and sh.top < 914400 * 0.5 and sh.left < 914400 * 2), None) or next(
        (sh.text_frame.text.split("\n")[0] for sh in sorted(s.shapes, key=lambda z: z.top) if sh.has_text_frame and len(sh.text_frame.text) > 8), "")
    head = {id(title): "Title", id(p1): "Part I cover: Forecasting Real Airfoil Wakes", id(p2): "Part II cover: What Does Pretraining Really Buy?",
            id(ty): "Thank You"}.get(id(s), head)
    md += f"\n## {k}. {head.strip()}\n\n{s.notes_slide.notes_text_frame.text.strip() if s.has_notes_slide else ''}\n"
open(NOTES_MD, "w").write(md)
print("wrote", OUT, N, "slides")
