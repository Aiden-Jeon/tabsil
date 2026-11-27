# /// script
# requires-python = ">=3.12"
# dependencies = ["fonttools", "uharfbuzz", "pillow"]
# ///
"""Generate the TabSil logo, icons, favicon and social preview in this folder.

The Hub model card banner is written to the card's images folder
(``training/hub/tabsil/images/``), which ``sync_hub_card.py`` uploads with the card.

The mark is a table grid with one thread weaving through its rows (*sil* is Korean
for thread). The wordmark is DM Sans Bold converted to outlines, so the SVGs need
no font at view time.

Usage (needs ``rsvg-convert`` on PATH, e.g. ``brew install librsvg``):

    curl -L -o /tmp/DMSans.ttf \
      "https://github.com/google/fonts/raw/main/ofl/dmsans/DMSans%5Bopsz,wght%5D.ttf"
    uv run assets/logo/make_assets.py --font /tmp/DMSans.ttf
"""

import argparse
import pathlib
import subprocess
import tempfile

import uharfbuzz as hb
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from fontTools.varLib.instancer import instantiateVariableFont
from PIL import Image

INK_LIGHT, INK_DARK, THREAD = "#1F2937", "#E6EDF3", "#E4572E"
MUTED_DARK, APP_BG, SOCIAL_BG = "#8B949E", "#111827", "#0D1117"
TAGLINE = "A tabular foundation model, woven through rows and columns"

# The thread enters at the top-left, turns at each row end and leaves at the
# bottom-right. TAILS covers the two places it crosses the frame, which get a gap.
THREAD_D = "M3 16H43a8 8 0 0 1 0 16H21a8 8 0 0 0 0 16H61"
TAILS_D = "M3 16H14M50 48H61"


def text_paths(font_path, text, size, x, y, wght, opsz):
    """Shape ``text`` with HarfBuzz and return one SVG path string per glyph, plus the advance."""
    font = instantiateVariableFont(TTFont(font_path), {"wght": wght, "opsz": opsz})
    glyphs, names = font.getGlyphSet(), font.getGlyphOrder()
    hb_font = hb.Font(hb.Face(pathlib.Path(font_path).read_bytes()))
    hb_font.set_variations({"wght": wght, "opsz": opsz})
    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf, {"kern": True, "liga": True})

    scale = size / font["head"].unitsPerEm
    pen_x, parts = 0, []
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions, strict=True):
        pen = SVGPathPen(glyphs)
        origin = (x + (pen_x + pos.x_offset) * scale, y - pos.y_offset * scale)
        glyphs[names[info.codepoint]].draw(TransformPen(pen, (scale, 0, 0, -scale, *origin)))
        parts.append(pen.getCommands())
        pen_x += pos.x_advance
    return parts, pen_x * scale


def mark(ink, uid, small=False):
    """The 64x64 mark. ``small`` drops the inner grid and thickens strokes for favicons."""
    frame_w, grid_w, thread_w, gap_w = (4, 0, 6, 11) if small else (3, 2, 4.5, 8.5)
    grid = (
        ""
        if small
        else (
            f'<path d="M8 24H56M8 40H56M24 8V56M40 8V56" fill="none" stroke="{ink}" '
            f'stroke-width="{grid_w}" stroke-opacity=".3"/>'
        )
    )
    return (
        f'<mask id="gap-{uid}" maskUnits="userSpaceOnUse" x="0" y="0" width="64" height="64">'
        f'<rect width="64" height="64" fill="#fff"/>'
        f'<path d="{TAILS_D}" fill="none" stroke="#000" stroke-width="{gap_w}"/></mask>'
        f'<rect mask="url(#gap-{uid})" x="8" y="8" width="48" height="48" rx="6" fill="none" '
        f'stroke="{ink}" stroke-width="{frame_w}"/>{grid}'
        f'<path d="{THREAD_D}" fill="none" stroke="{THREAD}" stroke-width="{thread_w}" '
        f'stroke-linecap="round"/>'
    )


def app_icon(uid, small=False):
    return (
        f'<rect width="64" height="64" rx="14" fill="{APP_BG}"/>'
        f'<g transform="translate(32 32) scale(.8) translate(-32 -32)">'
        f"{mark(INK_DARK, uid, small)}</g>"
    )


def lockup(font, ink, uid):
    """Mark plus the "TabSil" wordmark, "Sil" in the thread colour. Returns (body, width)."""
    glyphs, advance = text_paths(font, "TabSil", 38, 80, 44, wght=700, opsz=36)
    body = (
        mark(ink, uid)
        + f'<path fill="{ink}" d="{"".join(glyphs[:3])}"/>'
        + f'<path fill="{THREAD}" d="{"".join(glyphs[3:])}"/>'
    )
    return body, round(80 + advance + 4)


def svg(width, height, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">{body}</svg>\n'
    )


def build_svgs(font):
    files = {
        "tabsil-icon.svg": svg(64, 64, mark(INK_LIGHT, "l")),
        "tabsil-icon-dark.svg": svg(64, 64, mark(INK_DARK, "d")),
        "tabsil-icon-app.svg": svg(64, 64, app_icon("a")),
        "tabsil-icon-app-small.svg": svg(64, 64, app_icon("s", small=True)),
    }
    for name, ink in (("tabsil-logo.svg", INK_LIGHT), ("tabsil-logo-dark.svg", INK_DARK)):
        body, width = lockup(font, ink, name.removesuffix(".svg"))
        files[name] = svg(width, 64, body)

    body, width = lockup(font, INK_DARK, "social")
    scale = 4
    logo_x = (1280 - width * scale) / 2
    tagline, tagline_w = text_paths(font, TAGLINE, 30, 0, 0, wght=500, opsz=14)
    files["tabsil-social.svg"] = svg(
        1280,
        640,
        (
            f'<rect width="1280" height="640" fill="{SOCIAL_BG}"/>'
            f'<g transform="translate({logo_x:.1f} 150) scale({scale})">{body}</g>'
            f'<g transform="translate({(1280 - tagline_w) / 2:.1f} 478)">'
            f'<path fill="{MUTED_DARK}" d="{"".join(tagline)}"/></g>'
        ),
    )

    # Hub model card banner: a dark background of its own, so it reads in both Hub themes
    body, width = lockup(font, INK_DARK, "banner")
    scale = 3
    files["tabsil-banner.svg"] = svg(
        1280,
        320,
        (
            f'<rect width="1280" height="320" rx="24" fill="{SOCIAL_BG}"/>'
            f'<g transform="translate({(1280 - width * scale) / 2:.1f} 64) scale({scale})">'
            f"{body}</g>"
        ),
    )
    return files


def render(src, dst, *size_args):
    subprocess.run(["rsvg-convert", *size_args, str(src), "-o", str(dst)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--font", required=True, help="path to the DM Sans variable TTF")
    here = pathlib.Path(__file__).parent
    parser.add_argument("--out", type=pathlib.Path, default=here)
    parser.add_argument(
        "--card-images",
        type=pathlib.Path,
        default=here.resolve().parents[1] / "training" / "hub" / "tabsil" / "images",
        help="model card images folder that receives tabsil-banner.png",
    )
    args = parser.parse_args()

    for name, text in build_svgs(args.font).items():
        (args.out / name).write_text(text)

    render(args.out / "tabsil-icon-app.svg", args.out / "tabsil-icon-512.png", "-w", "512")
    render(args.out / "tabsil-logo.svg", args.out / "tabsil-logo.png", "-h", "128")
    render(args.out / "tabsil-logo-dark.svg", args.out / "tabsil-logo-dark.png", "-h", "128")
    render(args.out / "tabsil-social.svg", args.out / "tabsil-social.png")
    args.card_images.mkdir(parents=True, exist_ok=True)
    render(args.out / "tabsil-banner.svg", args.card_images / "tabsil-banner.png")
    with tempfile.TemporaryDirectory() as tmp:
        big = pathlib.Path(tmp) / "favicon-256.png"
        render(args.out / "tabsil-icon-app-small.svg", big, "-w", "256")
        Image.open(big).save(args.out / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])


if __name__ == "__main__":
    main()
