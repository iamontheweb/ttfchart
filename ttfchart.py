#!/usr/bin/env python3
"""Generate a PDF glyph chart for a TrueType/OpenType font.

Usage:
    uv run python ttfchart.py <font.ttf> [-o output.pdf]

Renders a "Font Information" page (name/version/license metadata plus a
text sample) followed by a grid of every glyph the font's cmap exposes,
grouped by Unicode block -- modelled on arial_unicode_google.pdf.

Color glyphs (COLR/CPAL, OT-SVG, CBDT/CBLC, sbix) are detected and drawn
as raster images so emoji-style fonts (e.g. Noto Color Emoji) render in
color instead of as blank/outline glyphs.

Generated with the assistance of Claude (Anthropic).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from collections import OrderedDict
from pathlib import Path

from fontTools import unicodedata as ft_unicodedata
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw
from PIL import ImageFont as PILImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont as RLTTFont
from reportlab.pdfgen import canvas

# ---------------------------------------------------------------------------
# Layout constants (tuned to resemble arial_unicode_google.pdf)
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = A4
MARGIN_L = 65
MARGIN_R = 65
MARGIN_TOP = 56
MARGIN_BOTTOM = 50
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
COLS = 14
COL_W = CONTENT_W / COLS
ROW_H = 44
GLYPH_FONT_SIZE = 20
GLYPH_IMG_PX = 128

BLUE = (48 / 255, 120 / 255, 182 / 255)
GRAY = (144 / 255, 144 / 255, 144 / 255)
LINE = (145 / 255, 181 / 255, 212 / 255)
BLACK = (0, 0, 0)

EXCLUDE_CATEGORIES = {"Cc", "Cf", "Cs"}

BASE_FONT = "Helvetica"
VECTOR_FONT_NAME = "TTFChartEmbedded"


# ---------------------------------------------------------------------------
# Font introspection
# ---------------------------------------------------------------------------

def get_name(font: TTFont, name_id: int) -> str:
    name_table = font["name"]
    rec = (
        name_table.getName(name_id, 3, 1, 0x409)
        or name_table.getName(name_id, 3, 0, 0x409)
        or name_table.getName(name_id, 1, 0, 0)
    )
    if rec is None:
        return ""
    try:
        return rec.toUnicode().strip()
    except Exception:
        return ""


def font_kind(font: TTFont) -> str:
    if "CFF2" in font or "CFF " in font:
        return "OpenType Font (PostScript outlines)"
    if "glyf" in font:
        return "True Type Font"
    return "Font"


def get_cmap(font: TTFont) -> dict[int, str]:
    try:
        return font.getBestCmap() or {}
    except Exception:
        return {}


def chartable_codepoints(cmap: dict[int, str]) -> list[int]:
    cps = []
    for cp in cmap:
        try:
            cat = unicodedata.category(chr(cp))
        except ValueError:
            cat = "Cn"
        if cat in EXCLUDE_CATEGORIES:
            continue
        cps.append(cp)
    return sorted(cps)


def group_by_block(cps: list[int]) -> "OrderedDict[str, list[int]]":
    groups: "OrderedDict[str, list[int]]" = OrderedDict()
    for cp in cps:
        try:
            block = ft_unicodedata.block(chr(cp))
        except Exception:
            block = "Unknown"
        groups.setdefault(block, []).append(cp)
    return groups


# ---------------------------------------------------------------------------
# Color glyph detection
# ---------------------------------------------------------------------------

def color_gids_from_svg(font: TTFont) -> set[int]:
    gids: set[int] = set()
    svg = font["SVG "]
    for doc in svg.docList:
        gids.update(range(doc.startGlyphID, doc.endGlyphID + 1))
    return gids


def color_gids_from_colr(font: TTFont) -> set[int]:
    gids: set[int] = set()
    colr = font["COLR"]
    try:
        if colr.version == 0:
            for glyph_name in colr.ColorLayers.keys():
                gids.add(font.getGlyphID(glyph_name))
        else:
            table = colr.table
            for rec in table.BaseGlyphList.BaseGlyphPaintRecord:
                gids.add(font.getGlyphID(rec.BaseGlyph))
    except Exception:
        pass
    return gids


def color_gids_from_bitmap(font: TTFont) -> set[int]:
    gids: set[int] = set()
    if "CBLC" in font:
        try:
            for strike in font["CBLC"].strikes:
                for subtable in strike.indexSubTables:
                    gids.update(
                        range(subtable.firstGlyphIndex, subtable.lastGlyphIndex + 1)
                    )
        except Exception:
            pass
    if "sbix" in font:
        try:
            for strike in font["sbix"].strikes.values():
                for glyph_name in strike.glyphs.keys():
                    gids.add(font.getGlyphID(glyph_name))
        except Exception:
            pass
    return gids


def detect_color(font: TTFont) -> tuple[bool, set[int]]:
    """Return (has_color_tables, set of color glyph ids)."""
    if "SVG " in font:
        return True, color_gids_from_svg(font)
    if "COLR" in font:
        return True, color_gids_from_colr(font)
    if "CBLC" in font or "sbix" in font:
        return True, color_gids_from_bitmap(font)
    return False, set()


# ---------------------------------------------------------------------------
# Color glyph rasterization
# ---------------------------------------------------------------------------

def rasterize_svg_glyphs(
    font: TTFont, gids_needed: set[int], px_size: int, workdir: Path
) -> dict[int, Path]:
    """Render OT-SVG glyphs to PNG files via Inkscape's batch shell mode."""
    inkscape = shutil.which("inkscape")
    results: dict[int, Path] = {}
    if not inkscape or not gids_needed:
        if gids_needed and not inkscape:
            print(
                "warning: 'inkscape' not found on PATH; OT-SVG color glyphs "
                "will be skipped. Install Inkscape to render them.",
                file=sys.stderr,
            )
        return results

    svg = font["SVG "]
    commands: list[str] = []
    for idx, doc in enumerate(svg.docList):
        relevant = sorted(
            g for g in range(doc.startGlyphID, doc.endGlyphID + 1) if g in gids_needed
        )
        if not relevant:
            continue
        data = doc.data
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        svg_path = workdir / f"doc{idx}.svg"
        svg_path.write_text(data, encoding="utf-8")
        commands.append(f"file-open:{svg_path}")
        for gid in relevant:
            out_path = workdir / f"g{gid}.png"
            commands += [
                f"export-id:glyph{gid}",
                "export-id-only",
                "export-type:png",
                f"export-width:{px_size}",
                f"export-height:{px_size}",
                f"export-filename:{out_path}",
                "export-do",
            ]
            results[gid] = out_path

    if not commands:
        return {}

    commands.append("quit")
    script = "\n".join(commands) + "\n"
    print(f"Rendering {len(results)} color glyph(s) with Inkscape...", file=sys.stderr)
    try:
        subprocess.run(
            [inkscape, "--shell"],
            input=script,
            text=True,
            capture_output=True,
            timeout=1800,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"warning: Inkscape rasterization failed: {exc}", file=sys.stderr)
        return {}

    return {gid: p for gid, p in results.items() if p.exists() and p.stat().st_size > 0}


def render_pillow_color(font_path: Path, cp: int, px_size: int) -> Image.Image | None:
    """Best-effort raster of a color glyph (COLRv0/CBDT/sbix) via Pillow/FreeType."""
    try:
        pad = max(4, px_size // 16)
        pil_font = PILImageFont.truetype(str(font_path), px_size)
        img = Image.new("RGBA", (px_size + pad * 2, px_size + pad * 2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((pad, pad), chr(cp), font=pil_font, embedded_color=True)
        bbox = img.getbbox()
        if bbox is None:
            return None
        return img.crop(bbox)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PDF drawing helpers
# ---------------------------------------------------------------------------

class Ctx:
    def __init__(self, c: canvas.Canvas):
        self.c = c
        self.y = PAGE_H - MARGIN_TOP
        self.pages = 1


def ensure_space(ctx: Ctx, needed: float) -> None:
    if ctx.y - needed < MARGIN_BOTTOM:
        ctx.c.showPage()
        ctx.y = PAGE_H - MARGIN_TOP
        ctx.pages += 1


def draw_rule(c: canvas.Canvas, y: float) -> None:
    c.setStrokeColorRGB(*LINE)
    c.setLineWidth(1)
    c.line(MARGIN_L, y, PAGE_W - MARGIN_R, y)


def wrap_text(text: str, font_name: str, size: float, max_width: float) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for word in words:
        candidate = f"{cur} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font_name, size) <= max_width:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def draw_info_page(
    ctx: Ctx, font: TTFont, sample_font_name: str, has_color: bool, cmap: dict[int, str]
) -> None:
    c = ctx.c
    x = MARGIN_L
    y = ctx.y

    c.setFillColorRGB(*BLUE)
    c.setFont(BASE_FONT, 24)
    c.drawString(x, y - 24, "Font Information")
    y -= 24 + 18

    c.setFillColorRGB(*BLACK)
    c.setFont(BASE_FONT, 11)
    line_h = 15

    full_name = get_name(font, 4) or get_name(font, 1)
    version = get_name(font, 5)
    designer = get_name(font, 9) or get_name(font, 8)
    copyright_ = get_name(font, 0)
    license_desc = get_name(font, 13)

    lines: list[str] = []
    if full_name:
        lines.append(full_name)
    if version:
        lines.append(version)
    lines.append(font_kind(font))
    if has_color:
        lines.append("Contains color glyphs")
    if designer:
        lines.append(f"Creator: {designer}")

    for line in lines:
        c.drawString(x, y - 11, line)
        y -= line_h

    if copyright_:
        for wrapped in wrap_text(copyright_, BASE_FONT, 11, CONTENT_W):
            c.drawString(x, y - 11, wrapped)
            y -= line_h

    if license_desc:
        y -= line_h
        for wrapped in wrap_text(license_desc, BASE_FONT, 11, CONTENT_W):
            c.drawString(x, y - 11, wrapped)
            y -= line_h

    y -= 8
    draw_rule(c, y)
    y -= 30

    c.setFillColorRGB(*BLUE)
    c.setFont(BASE_FONT, 14)
    c.drawString(x, y - 14, "Text Sample")
    y -= 14 + 30

    pangram_upper = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    pangram_lower = "the quick brown fox jumps over the lazy dog"
    sample_size = 22

    has_basic_latin = all(ord(ch) in cmap for ch in "ABCXYZabcxyz")
    active_sample_font = sample_font_name if has_basic_latin else BASE_FONT

    c.setFillColorRGB(*BLACK)
    for pangram in (pangram_upper, pangram_lower):
        used_font = active_sample_font
        try:
            width = pdfmetrics.stringWidth(pangram, used_font, sample_size)
        except Exception:
            used_font = BASE_FONT
            width = pdfmetrics.stringWidth(pangram, used_font, sample_size)
        size = sample_size
        if width > CONTENT_W:
            size *= CONTENT_W / width
        c.setFont(used_font, size)
        c.drawCentredString(PAGE_W / 2, y - size, pangram)
        y -= size + 34

    ctx.y = y


def draw_heading(ctx: Ctx, text: str) -> None:
    c = ctx.c
    ensure_space(ctx, 40)
    draw_rule(c, ctx.y)
    ctx.y -= 24
    c.setFillColorRGB(*BLUE)
    c.setFont(BASE_FONT, 13)
    c.drawString(MARGIN_L, ctx.y - 13, text)
    ctx.y -= 13 + 16


def draw_vector_glyph(c: canvas.Canvas, font_name: str, cp: int, cx: float, y_bot: float) -> None:
    try:
        c.setFont(font_name, GLYPH_FONT_SIZE)
    except Exception:
        return
    c.setFillColorRGB(*BLACK)
    baseline = y_bot + ROW_H * 0.32
    try:
        c.drawCentredString(cx + COL_W / 2, baseline, chr(cp))
    except Exception:
        pass


def draw_color_glyph(c: canvas.Canvas, image_src, cx: float, y_bot: float) -> None:
    reader = ImageReader(image_src if not isinstance(image_src, Path) else str(image_src))
    iw, ih = reader.getSize()
    label_h = 14
    avail_w = COL_W - 8
    avail_h = ROW_H - label_h - 4
    if iw <= 0 or ih <= 0:
        return
    scale = min(avail_w / iw, avail_h / ih)
    w, h = iw * scale, ih * scale
    x = cx + (COL_W - w) / 2
    y = y_bot + (avail_h - h) / 2 + 2
    c.drawImage(reader, x, y, w, h, mask="auto")


def draw_grid(
    ctx: Ctx,
    cps: list[int],
    color_images: dict[int, object],
    vector_font_name: str | None,
) -> None:
    c = ctx.c
    i = 0
    n = len(cps)
    while i < n:
        row = cps[i : i + COLS]
        ensure_space(ctx, ROW_H)
        y_top = ctx.y
        y_bot = y_top - ROW_H
        for col, cp in enumerate(row):
            cx = MARGIN_L + col * COL_W
            c.setStrokeColorRGB(*LINE)
            c.setLineWidth(0.6)
            c.rect(cx, y_bot, COL_W, ROW_H, stroke=1, fill=0)

            c.setFillColorRGB(*GRAY)
            c.setFont(BASE_FONT, 7.5)
            c.drawString(cx + 4, y_top - 11, format(cp, "04X"))

            if cp in color_images:
                draw_color_glyph(c, color_images[cp], cx, y_bot)
            elif vector_font_name:
                draw_vector_glyph(c, vector_font_name, cp, cx, y_bot)
        ctx.y = y_bot
        i += COLS


def draw_glyph_sections(
    ctx: Ctx,
    groups: "OrderedDict[str, list[int]]",
    color_images: dict[int, object],
    vector_font_name: str | None,
) -> None:
    for block_name, cps in groups.items():
        draw_heading(ctx, block_name)
        draw_grid(ctx, cps, color_images, vector_font_name)
        ctx.y -= 20


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_pdf(font_path: Path, output_path: Path) -> None:
    font = TTFont(font_path, lazy=True)
    cmap = get_cmap(font)
    if not cmap:
        raise SystemExit(f"error: {font_path} has no usable cmap (no character mapping found)")

    cps = chartable_codepoints(cmap)
    groups = group_by_block(cps)
    cp_gid = {cp: font.getGlyphID(cmap[cp]) for cp in cps}

    has_color, color_gids = detect_color(font)
    color_cps = {cp for cp, gid in cp_gid.items() if gid in color_gids} if color_gids else set()

    color_images: dict[int, object] = {}
    tmpdir: Path | None = None

    if color_cps:
        if "SVG " in font:
            tmpdir = Path(tempfile.mkdtemp(prefix="ttfchart_svg_"))
            needed_gids = {cp_gid[cp] for cp in color_cps}
            gid_to_png = rasterize_svg_glyphs(font, needed_gids, GLYPH_IMG_PX, tmpdir)
            for cp in list(color_cps):
                png = gid_to_png.get(cp_gid[cp])
                if png is not None:
                    color_images[cp] = png
                else:
                    color_cps.discard(cp)
        else:
            print(f"Rendering {len(color_cps)} color glyph(s)...", file=sys.stderr)
            for cp in list(color_cps):
                img = render_pillow_color(font_path, cp, GLYPH_IMG_PX)
                if img is not None:
                    color_images[cp] = img
                else:
                    color_cps.discard(cp)

    vector_font_name: str | None = VECTOR_FONT_NAME
    try:
        pdfmetrics.registerFont(RLTTFont(VECTOR_FONT_NAME, str(font_path)))
    except Exception as exc:
        print(f"warning: could not register font for vector rendering: {exc}", file=sys.stderr)
        vector_font_name = None

    c = canvas.Canvas(str(output_path), pagesize=A4)
    ctx = Ctx(c)

    sample_font_name = vector_font_name or BASE_FONT
    draw_info_page(ctx, font, sample_font_name, has_color, cmap)
    draw_glyph_sections(ctx, groups, color_images, vector_font_name)

    c.save()

    if tmpdir is not None:
        shutil.rmtree(tmpdir, ignore_errors=True)

    total = len(cps)
    print(
        f"Wrote {output_path} ({total} glyphs, {len(color_images)} color) "
        f"across {ctx.pages} page(s).",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("font", type=Path, help="Path to a .ttf/.otf font file")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output PDF path (default: <font>.pdf)"
    )
    args = parser.parse_args()

    if not args.font.exists():
        raise SystemExit(f"error: font file not found: {args.font}")

    output = args.output or Path.cwd() / args.font.with_suffix(".pdf").name
    build_pdf(args.font, output)


if __name__ == "__main__":
    main()
