# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Generate the README charts (the SVGs next to this script) from the result tables.

The numbers are read from the Markdown tables of ``README.md`` and ``docs/task-results.md``, which
stay the single source of truth: a chart cannot disagree with its table. Edit the table, then run
this script again. Every chart is written twice, ``<name>-light.svg`` and ``<name>-dark.svg``, for
the README's ``<picture>`` (the same pattern as the logo).

TabSil is the accent color (the logo's thread) and every other method is gray: one series is the
point, the rest is context. Text is live (a system sans stack) so the SVGs carry no font files; the
layout leaves room for wider fonts than the one it was checked with.

Usage (standard library only):

    uv run assets/charts/make_charts.py            # write assets/charts/*.svg
    uv run assets/charts/make_charts.py --check    # exit 1 if the committed SVGs are out of date
"""

import argparse
import math
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
README = ROOT / "README.md"
TASK_DOC = ROOT / "docs" / "task-results.md"
OUT = HERE

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
MINUS = "−"

# Colors per mode. `bg` is the page behind the transparent SVG (GitHub's), used for the 2px rings
# around marks. Marks keep >= 3:1 against it and the accent differs from the gray in lightness too
# (OKLCH L gap >= 0.08), so the emphasis survives grayscale and color-blind viewing.
THEMES = {
    "light": {
        "bg": "#FFFFFF",
        "ink": "#1F2937",
        "ink2": "#4B5563",
        "muted": "#6B7280",
        "grid": "#E5E7EB",
        "axis": "#D1D5DB",
        "other": "#6B7280",
        "accent": "#E4572E",
        "band": 0.09,
    },
    "dark": {
        "bg": "#0D1117",
        "ink": "#E6EDF3",
        "ink2": "#B1BAC4",
        "muted": "#8B949E",
        "grid": "#21262D",
        "axis": "#30363D",
        "other": "#7D8590",
        "accent": "#F5784F",
        "band": 0.16,
    },
}


# -- reading the tables -----------------------------------------------------------------------


def read_tables(path: Path) -> dict[str, list[list[str]]]:
    """The first Markdown table under each heading: a header row, then the data rows.

    Bold markers and backticks are dropped from the cells. Lines inside code fences are skipped.
    """
    found: dict[str, list[list[str]]] = {}
    heading, block, in_code = "", [], False

    def flush() -> None:
        rows = [r for r in block if not all(re.fullmatch(r":?-{3,}:?", c) for c in r)]
        if rows and heading not in found:
            found[heading] = rows
        block.clear()

    for line in path.read_text().splitlines():
        if line.startswith("```"):
            in_code = not in_code
        elif in_code:
            continue
        elif line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
        elif line.lstrip().startswith("|"):
            cells = line.strip().strip("|").split("|")
            block.append([c.strip().replace("**", "").replace("`", "") for c in cells])
        else:
            flush()
    flush()
    return found


def table(path: Path, heading: str, first_header: str) -> list[list[str]]:
    rows = read_tables(path).get(heading)
    if not rows or rows[0][0] != first_header:
        raise SystemExit(f"{path.name}: no table '{first_header} | ...' under '{heading}'")
    return rows


def elo_ci(cell: str) -> tuple[float, float, float]:
    """'1755 (+88/-73)' -> (1755, 88, 73)."""
    m = re.fullmatch(rf"(\d+) \(\+(\d+)/[{MINUS}-](\d+)\)", cell)
    if not m:
        raise SystemExit(f"not 'Elo (+up/-down)': {cell!r}")
    return float(m[1]), float(m[2]), float(m[3])


def mean_se(cell: str) -> tuple[float, float]:
    """'0.55 ± 0.09' -> (0.55, 0.09)."""
    m = re.fullmatch(r"(\d+(?:\.\d+)?) ± (\d+(?:\.\d+)?)", cell)
    if not m:
        raise SystemExit(f"not 'mean ± se': {cell!r}")
    return float(m[1]), float(m[2])


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def num(v: float) -> str:
    """A short number for SVG attributes."""
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


# -- a tiny SVG writer ------------------------------------------------------------------------


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    """Collects SVG elements; colors come from the theme."""

    def __init__(self, width: int, height: int, mode: str, title: str, desc: str):
        self.width, self.height, self.mode = width, height, mode
        self.c = THEMES[mode]
        self.title, self.desc = title, desc
        self.items: list[str] = []

    def line(self, x1, y1, x2, y2, color, width=1.0, cap="butt"):
        self.items.append(
            f'<line x1="{num(x1)}" y1="{num(y1)}" x2="{num(x2)}" y2="{num(y2)}" '
            f'stroke="{color}" stroke-width="{num(width)}" stroke-linecap="{cap}"/>'
        )

    def rect(self, x, y, w, h, fill, rx=0.0, opacity=None):
        extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.items.append(
            f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}" '
            f'rx="{num(rx)}" fill="{fill}"{extra}/>'
        )

    def circle(self, x, y, r, fill, stroke=None, stroke_width=0.0):
        extra = f' stroke="{stroke}" stroke-width="{num(stroke_width)}"' if stroke else ""
        self.items.append(
            f'<circle cx="{num(x)}" cy="{num(y)}" r="{num(r)}" fill="{fill}"{extra}/>'
        )

    def dot(self, x, y, r, color, hollow=False):
        """A marker with a 2px ring in the page color, so it stays legible over lines."""
        bg = self.c["bg"]
        self.circle(x, y, r + 2, bg)
        if hollow:
            self.circle(x, y, r - 1, bg, stroke=color, stroke_width=2)
        else:
            self.circle(x, y, r, color)

    def text(self, x, y, s, size=12, color=None, weight=400, anchor="start", tabular=False):
        """Text with its baseline at ``y``."""
        color = color or self.c["ink"]
        extra = ' font-variant-numeric="tabular-nums"' if tabular else ""
        weight_attr = f' font-weight="{weight}"' if weight != 400 else ""
        anchor_attr = f' text-anchor="{anchor}"' if anchor != "start" else ""
        self.items.append(
            f'<text x="{num(x)}" y="{num(y)}" font-size="{num(size)}" fill="{color}"'
            f"{weight_attr}{anchor_attr}{extra}>{esc(s)}</text>"
        )

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}" role="img" aria-labelledby="title desc" '
            f"font-family='{FONT}'>\n"
            f'<title id="title">{esc(self.title)}</title>\n<desc id="desc">{esc(self.desc)}</desc>'
        )
        return head + "\n" + "\n".join(self.items) + "\n</svg>\n"


# -- chart 1: TabArena-Lite Elo ---------------------------------------------------------------


def tabarena_lite(mode: str) -> str:
    """Elo with 95% CI of the README's TabArena-Lite table, TabSil highlighted."""
    readme = README.read_text()
    rows = table(README, "Results", "Method")
    place_of_pool = re.search(r"places (\d+)\w+ of (\d+) methods", readme)
    datasets = re.search(r"TabArena-Lite has (\d+) datasets", readme)
    if not (place_of_pool and datasets):
        raise SystemExit("README.md: the TabArena-Lite sentences changed; update make_charts.py")
    pool = int(place_of_pool[2])

    data = []
    for r in rows[1:]:
        elo, up, down = elo_ci(r[2])
        data.append(
            {"name": r[0], "place": int(r[1]), "elo": elo, "lo": elo - down, "hi": elo + up}
        )
    focal = next(d for d in data if d["name"] == "TabSil")

    svg_probe = Svg(1, 1, mode, "", "")
    c = svg_probe.c
    width, pitch, top = 840, 26, 92
    x0, x1 = 300, 796
    lo_v, hi_v = 900, 2300
    n = len(data)
    axis_y = top + (n - 1) * pitch + pitch / 2 + 8
    height = int(axis_y + 58)

    def x_of(v: float) -> float:
        return x0 + (v - lo_v) / (hi_v - lo_v) * (x1 - x0)

    title = "TabArena-Lite Elo"
    desc = (
        f"Elo with 95% confidence intervals on TabArena-Lite ({datasets[1]} datasets, official "
        f"pipeline, 8-fold bagged), RandomForest (default) = 1000. "
        + "; ".join(f"{d['name']} {d['elo']:.0f} (place {d['place']})" for d in data)
        + f". TabSil places {ordinal(focal['place'])} of {pool} methods."
    )
    svg = Svg(width, height, mode, title, desc)

    svg.text(24, 32, title, 17, c["ink"], 600)
    svg.text(
        24,
        54,
        f"{datasets[1]} datasets · {pool} methods in the pool · dots are Elo, whiskers the 95% CI"
        " · higher is better",
        12,
        c["ink2"],
    )

    # the TabSil row gets a faint band behind everything
    fy = top + data.index(focal) * pitch
    svg.rect(12, fy - pitch / 2 + 1, width - 24, pitch - 2, c["accent"], rx=6, opacity=c["band"])

    for v in range(1000, 2300, 250):
        svg.line(x_of(v), top - pitch / 2, x_of(v), axis_y, c["grid"])
        svg.text(x_of(v), axis_y + 20, str(v), 11, c["muted"], anchor="middle", tabular=True)
    svg.line(x0, axis_y, x1, axis_y, c["axis"])
    svg.text(
        (x0 + x1) / 2,
        axis_y + 42,
        "Elo (RandomForest (default) = 1000)",
        12,
        c["ink2"],
        anchor="middle",
    )
    svg.text(46, top - pitch / 2 - 6, "place", 11, c["muted"], anchor="end")

    for i, d in enumerate(data):
        cy = top + i * pitch
        is_focal = d is focal
        color = c["accent"] if is_focal else c["other"]
        # a hairline from the label column to the plot leads the eye along the row
        svg.line(276, cy, x1, cy, c["grid"])
        svg.text(46, cy + 4, str(d["place"]), 11, c["muted"], anchor="end", tabular=True)
        svg.text(60, cy + 4.5, d["name"], 12.5, c["ink"], 700 if is_focal else 400)
        svg.line(x_of(d["lo"]), cy, x_of(d["hi"]), cy, color, 2, "round")
        svg.dot(x_of(d["elo"]), cy, 6 if is_focal else 4.5, color)

    # selective value labels: the story (TabSil), the top and the anchor
    for d, label, weight in (
        (focal, f"{focal['elo']:.0f} · {ordinal(focal['place'])} of {pool}", 700),
        (data[0], f"{data[0]['elo']:.0f}", 400),
        (data[-1], f"{data[-1]['elo']:.0f}", 400),
    ):
        cy = top + data.index(d) * pitch
        svg.text(x_of(d["hi"]) + 10, cy + 4, label, 12, c["ink"], weight, tabular=True)

    return svg.render()


# -- chart 2: the task suites -----------------------------------------------------------------

SHORT_NAMES = {
    "TabPFN-TS-3 (public result)": "TabPFN-TS-3",
    "TabPFN-TS (TabPFN v2)": "TabPFN-TS",
    "AutoARIMA (public result)": "AutoARIMA",
    "TabSil S-learner (default)": "TabSil S-learner",
    "EconML T-learner (GBM)": "EconML T-learner",
    "EconML X-learner (GBM)": "EconML X-learner",
    "EconML S-learner (GBM)": "EconML S-learner",
    "EconML DR-learner (GBM)": "EconML DR-learner",
    "EconML CausalForestDML": "EconML CausalForest",
    "TabPFN v2 extension, simple_impute": "TabPFN simple_impute",
    "TabPFN v2 extension, impute": "TabPFN impute",
}


def lin_scale(v0: float, v1: float, p0: float, p1: float):
    return lambda v: p0 + (v - v0) / (v1 - v0) * (p1 - p0)


def log_scale(v0: float, v1: float, p0: float, p1: float):
    a, b = math.log(v0), math.log(v1)
    return lambda v: p0 + (math.log(v) - a) / (b - a) * (p1 - p0)


def dotplot(svg: Svg, x: float, y: float, labels: list[str], plots: list[dict], pitch: int = 21):
    """Row labels on the left, one or more plots to their right sharing the rows.

    ``y`` is the center of the first row. A plot has the pixel range ``x0``-``x1``, a ``scale``,
    ``ticks`` with a formatter ``fmt``, an optional ``title`` and one entry per row in ``rows``:
    ``span`` (a whisker or dumbbell segment in data units), ``marks`` as ``(value, hollow)`` and
    an optional ``tag`` printed after the row. Rows named TabSil get the accent color. Returns the
    y of the axis line.
    """
    c = svg.c
    centers = [y + i * pitch for i in range(len(labels))]
    axis_y = centers[-1] + pitch / 2 + 2
    right = max(p["x1"] for p in plots)
    focal = [label.startswith("TabSil") for label in labels]

    for cy, is_focal in zip(centers, focal, strict=True):
        if is_focal:
            svg.rect(
                x - 8,
                cy - pitch / 2 + 1,
                right - x + 16,
                pitch - 2,
                c["accent"],
                rx=5,
                opacity=c["band"],
            )
    for cy, label, is_focal in zip(centers, labels, focal, strict=True):
        svg.text(x, cy + 4, label, 11.5, c["ink"], 700 if is_focal else 400)

    for p in plots:
        scale = p["scale"]
        if p.get("title"):
            svg.text(
                (p["x0"] + p["x1"]) / 2,
                y - pitch / 2 - 7,
                p["title"],
                11,
                c["ink2"],
                600,
                anchor="middle",
            )
        for t in p["ticks"]:
            svg.line(scale(t), y - pitch / 2, scale(t), axis_y, c["grid"])
            svg.text(
                scale(t), axis_y + 15, p["fmt"](t), 10.5, c["muted"], anchor="middle", tabular=True
            )
        svg.line(p["x0"], axis_y, p["x1"], axis_y, c["axis"])
        for cy, row, is_focal in zip(centers, p["rows"], focal, strict=True):
            color = c["accent"] if is_focal else c["other"]
            svg.line(p["x0"], cy, p["x1"], cy, c["grid"])
            if row.get("span"):
                svg.line(scale(row["span"][0]), cy, scale(row["span"][1]), cy, color, 2, "round")
            for value, hollow in row["marks"]:
                svg.dot(scale(value), cy, 5.5 if is_focal else 4.5, color, hollow=hollow)
            if row.get("tag"):
                end = max([v for v, _ in row["marks"]] + list(row.get("span") or ()))
                svg.text(scale(end) + 10, cy + 4, row["tag"], 11, c["ink"], 700, tabular=True)
    return axis_y


def panel_header(svg: Svg, x: float, y: float, title: str, hint: str) -> None:
    svg.text(x, y + 14, title, 13.5, svg.c["ink"], 600)
    svg.text(x, y + 32, hint, 11.5, svg.c["ink2"])


def other_tasks(mode: str) -> str:
    """Four panels, one per task suite, from the tables of docs/task-results.md."""
    doc = TASK_DOC.read_text()
    place = re.search(r"TabSil places (\d+)\w+ of (\d+)", doc)
    if not place:
        raise SystemExit("docs/task-results.md: the ADBench place sentence changed")

    # A: forecasting, SQL Elo with 95% CI
    name_a = "Forecasting (fev-bench)"
    fc = table(TASK_DOC, name_a, "Method")
    a_labels, a_rows, a_desc = [], [], ""
    for r in fc[1:]:
        elo, up, down = elo_ci(r[1])
        a_labels.append(SHORT_NAMES.get(r[0], r[0]))
        a_rows.append({"span": (elo - down, elo + up), "marks": [(elo, False)]})
        if r[0] == "TabSil":
            a_rows[-1]["tag"] = f"{elo:.0f}"
            a_desc = f"SQL Elo {elo:.0f}, place {len(a_rows)} of {len(fc) - 1}"

    # B: outlier detection, mean AUROC
    name_b = "Outlier detection (ADBench)"
    ad = table(TASK_DOC, name_b, "Method")
    b_labels, b_rows, b_desc = [], [], ""
    for r in ad[1:]:
        auroc = float(r[1])
        b_labels.append(SHORT_NAMES.get(r[0], r[0]))
        b_rows.append({"marks": [(auroc, False)]})
        if r[0] == "TabSil":
            b_rows[-1]["tag"] = r[1]
            b_desc = f"mean AUROC {r[1]}, {ordinal(int(place[1]))} of {place[2]} by Elo"

    # C: treatment effects, out-of-sample sqrt(PEHE) +- s.e. on IHDP and ACIC 2016
    name_c = "Treatment effects (IHDP, ACIC 2016)"
    ce = table(TASK_DOC, name_c, "Method")
    c_labels, c_ihdp, c_acic, c_desc = [], [], [], ""
    for r in ce[1:]:
        c_labels.append(SHORT_NAMES.get(r[0], r[0]))
        rows_for = []
        for col in (2, 5):  # IHDP and ACIC 2016 out-of-sample
            mean, se = mean_se(r[col])
            rows_for.append({"span": (mean - se, mean + se), "marks": [(mean, False)]})
        c_ihdp.append(rows_for[0])
        c_acic.append(rows_for[1])
        if r[0].startswith("TabSil S-learner"):
            c_ihdp[-1]["tag"], c_acic[-1]["tag"] = r[2].split(" ± ")[0], r[5].split(" ± ")[0]
            c_desc = f"S-learner out-of-sample sqrt(PEHE) {r[2]} on IHDP and {r[5]} on ACIC 2016"

    # D: imputation, NRMSE with 10% and 50% of the cells masked
    name_d = "Imputation (OpenML, MCAR)"
    im = table(TASK_DOC, name_d, "Method")
    d_labels, d_rows, d_desc = [], [], ""
    for r in im[1:]:
        lo, hi = float(r[1]), float(r[3])
        d_labels.append(SHORT_NAMES.get(r[0], r[0]))
        d_rows.append({"span": (min(lo, hi), max(lo, hi)), "marks": [(lo, True), (hi, False)]})
        if r[0] == "TabSil":
            d_rows[-1]["tag"] = f"{r[1]} → {r[3]}"
            d_desc = f"NRMSE {r[1]} at 10% to {r[3]} at 50% masked"

    width, pad, col_w, gap = 840, 24, 372, 48
    x_left, x_right = pad, pad + col_w + gap
    pitch, head = 21, 46
    row1_y, row2_y = 22, 22 + head + 10 * pitch + 34 + 30
    height = row2_y + head + 7 * pitch + 34 + 44

    title = "TabSil on other tasks, zero-shot"
    desc = (
        "Four panels, TabSil highlighted against the baselines of each benchmark. "
        f"Forecasting (fev-bench): {a_desc}. Outlier detection (ADBench): {b_desc}. "
        f"Treatment effects: {c_desc}. Imputation (OpenML, MCAR): {d_desc}."
    )
    svg = Svg(width, height, mode, title, desc)
    c = svg.c

    # A: forecasting
    panel_header(svg, x_left, row1_y, name_a, "SQL Elo with 95% CI · higher is better")
    ax0, ax1 = x_left + 112, x_left + col_w - 40
    dotplot(
        svg,
        x_left,
        row1_y + head + 8,
        a_labels,
        [
            {
                "x0": ax0,
                "x1": ax1,
                "rows": a_rows,
                "ticks": (1000, 1500, 2000),
                "scale": lin_scale(700, 2100, ax0, ax1),
                "fmt": lambda v: f"{v:.0f}",
            }
        ],
    )

    # C: treatment effects, two plots sharing the rows
    panel_header(
        svg, x_right, row1_y, name_c, "out-of-sample √PEHE ± s.e. · log scale · lower is better"
    )
    label_w = 134
    pw = (col_w - label_w - 14) / 2
    plots_c = []
    for k, (title_k, rows_k, v0, v1, ticks) in enumerate(
        (
            ("IHDP", c_ihdp, 0.4, 15, (0.5, 1, 2, 5, 10)),
            ("ACIC 2016", c_acic, 0.3, 1.6, (0.5, 1)),
        )
    ):
        a0 = x_right + label_w + k * (pw + 14)
        plots_c.append(
            {
                "x0": a0,
                "x1": a0 + pw,
                "rows": rows_k,
                "ticks": ticks,
                "title": title_k,
                "scale": log_scale(v0, v1, a0, a0 + pw),
                "fmt": lambda v: f"{v:g}",
            }
        )
    dotplot(svg, x_right, row1_y + head + 8 + 14, c_labels, plots_c)

    # B: outlier detection
    panel_header(svg, x_left, row2_y, name_b, "mean AUROC (0.5 is chance) · higher is better")
    bx0, bx1 = x_left + 134, x_left + col_w - 40
    axis_b = dotplot(
        svg,
        x_left,
        row2_y + head + 8,
        b_labels,
        [
            {
                "x0": bx0,
                "x1": bx1,
                "rows": b_rows,
                "ticks": (0.5, 0.6, 0.7, 0.8),
                "scale": lin_scale(0.5, 0.8, bx0, bx1),
                "fmt": lambda v: f"{v:.1f}",
            }
        ],
    )
    note = f"TabSil places {ordinal(int(place[1]))} of {place[2]} methods by Elo."
    svg.text(x_left, axis_b + 40, note, 11, c["muted"])
    svg.text(
        x_left, axis_b + 56, "Shown: the three highest-Elo methods and TabSil.", 11, c["muted"]
    )

    # D: imputation
    panel_header(
        svg, x_right, row2_y, name_d, "NRMSE at 10% and 50% of cells masked · lower is better"
    )
    dx0, dx1 = x_right + 142, x_right + col_w - 10
    axis_d = dotplot(
        svg,
        x_right,
        row2_y + head + 8,
        d_labels,
        [
            {
                "x0": dx0,
                "x1": dx1,
                "rows": d_rows,
                "ticks": (0.5, 1, 2),
                "scale": log_scale(0.5, 3, dx0, dx1),
                "fmt": lambda v: f"{v:g}",
            }
        ],
    )
    svg.dot(dx0 + 5, axis_d + 36, 4.5, c["muted"], hollow=True)
    svg.text(dx0 + 16, axis_d + 40, "10% masked", 11, c["ink2"])
    svg.dot(dx0 + 104, axis_d + 36, 4.5, c["muted"])
    svg.text(dx0 + 115, axis_d + 40, "50% masked", 11, c["ink2"])

    return svg.render()


CHARTS = {
    "tabarena-lite": tabarena_lite,
    "other-tasks": other_tasks,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="fail if the SVGs on disk differ")
    args = parser.parse_args()

    stale = []
    for name, build in CHARTS.items():
        for mode in THEMES:
            path = OUT / f"{name}-{mode}.svg"
            svg = build(mode)
            if args.check:
                if not path.exists() or path.read_text() != svg:
                    stale.append(path.name)
            else:
                path.write_text(svg)
                print(f"wrote {path.relative_to(ROOT)} ({len(svg) / 1024:.1f} KB)")
    if stale:
        print("out of date (run `uv run assets/charts/make_charts.py`):", ", ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
