"""Real-table corpus for the continued-training screen (v2 top-5 plan, items 1.2 and 2.2 of
`training/docs/v2-top5-plan.md`): OpenML classification and regression tables that share no
dataset with the benchmarks the project evaluates on, and the leakage record that decides it.

Inputs (public downloads only; no credentials, no model, no GPU):
- OpenML REST API (api.openml.org): the candidates and the OpenML-backed benchmark datasets
  (descriptions, features, qualities). The rows come from the parquet file of each description
  (`parquet_url`), or from the ARFF file when there is none.
- TabArena's 51 datasets: `curated_tabarena_dataset_metadata.csv` of autogluon/tabarena at commit
  8e55fb9 (GitHub raw), which gives the OpenML dataset ids.
- The development suite: the 57 AMLB and OpenML-CTR23 tasks of `benchmarks/tabular/amlb.py`
  (`PAPER_CLASSIFICATION`, `PAPER_REGRESSION`; `TABARENA_OVERLAP` marks the 9 also in TabArena).
- BeyondArena's 142 datasets: `BeyondArena_tasks_metadata.csv` of the same commit (names, shapes,
  targets), and the Hugging Face dataset `TabArena/BeyondArena` at revision 2ecfe88 for the rows:
  each parquet is read as a footer for its shape and names, and hashed up to `--max-benchmark-mb`.
  Datasets without a parquet in that repo keep their name and shape only.
- ADBench's 12 tables (`benchmarks/unsupervised/adbench.py`: `DATASETS`, `SHA256`; the npz files at
  its commit; the npz has no feature names) and the 10 imputation tables (`unsupervised/impute.py`:
  `TABLES`, OpenML ids).

Candidate filters, before the leakage check (a failing candidate is recorded in excluded.csv):
active; a default target; 500 to 100,000 rows; 2 to 100 predictors; a classification target with
2 to 10 classes or a numeric regression target; at most 50% missing predictor cells; no time-series,
image, text or forecasting tag or name (and not of the FOREX or Meta-Album families: price series
and image features); not drawn from a known generator (step "synthetic": name families such as
Friedman's, Delve's and the F16 simulators, BNG, LED, GAMETES, parity, MONK's problems and
Monte Carlo samples, or an OpenML description with a word starting "synthetic", "artificial" or
"simulat", or the words "generated" or "generator"). Candidates are
taken in OpenML id order and the first `--max-candidates` that pass are checked.

Leakage rules, in this order. The first rule a candidate fails excludes it, and the matched
benchmark dataset is recorded.
1. The same OpenML dataset id as a benchmark dataset, or a normalized name equal to a benchmark's
   name (lower case; separators and version suffixes removed).
2. The same shape (rows within 1% of the benchmark's rows, the same number of predictors) and a
   predictor-name Jaccard similarity of at least 0.5.
3. A predictor-name Jaccard similarity of at least 0.8.
4. After download, a row hash shared with a benchmark dataset's rows for more than 1% of the
   candidate's rows. Two hashes per row: with names (column names and values, columns in name
   order) and with names ignored (values sorted). Numbers have 6 significant digits, missing
   cells a marker. A benchmark without feature names (ADBench) has only the names-ignored hash.
5. Duplicates among the candidates, keeping the lowest id: the same name stem and row count
   (variants of one source, before download), the same normalized name, then a shared row hash for
   more than 1% of the rows (both checked after rules 1 to 4 and 7).
6. A row count within 1% of a benchmark dataset's, whatever the columns: a copy of a benchmark's
   rows with other columns, an encoding or names of its own (e.g. a one-hot or renamed copy). It
   excludes many unrelated tables too; that is accepted.
7. After download, columns shared with one benchmark dataset, compared without names, category
   labels or row order. A column with 2 to `LOW_CARD` distinct values is compared by its category
   shares, sorted (L1 distance at most `SHARES_L1`, the same number of categories; columns of 3
   or more categories count). A column with more distinct values (`MIN_DISTINCT`, or
   `MIN_DISTINCT_INT` for integer values) is compared by its values: as they are, and for numbers
   min-max normalized, so that a scaled or standardized copy matches; it matches a benchmark column
   that holds at least `CONTAINED` of its values (estimated on the `SAMPLE` smallest value hashes
   of each benchmark column). Excluded: two value columns that match one benchmark dataset, or at
   least 3 matching columns that are at least 60% of the compared ones. This finds copies whose
   rows rule 4 cannot match, because the benchmark's copy (TabArena's curated tables) labels its
   categories or names its columns otherwise, or holds other rows of the same data.

Outputs in `--out`: manifest.csv (kept tables), excluded.csv (every exclusion with its step and the
matched benchmark dataset), candidates.csv (every candidate examined, with its status),
benchmarks.csv (the benchmark side, with whether feature names and row hashes were available),
tables/<id>.parquet (the kept tables) and summary.md (counts per step, what could not be checked).

From the repository root (CPU only; it downloads and hashes, it trains nothing):
  uv run --frozen --package tabsil-train python -I training/scripts/real_corpus.py \\
    --cache /tmp/top5-corpus-data --out /tmp/top5-corpus-out
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import arff
import numpy as np
import pandas as pd

from tabsil_train.benchmarks import _files
from tabsil_train.benchmarks.tabular import amlb
from tabsil_train.benchmarks.unsupervised import adbench, impute

OPENML = "https://api.openml.org/api/v1/json"
OPENML_FILE = "https://api.openml.org/data/v1/download/{file_id}"
TABARENA_SHA = "8e55fb9fd20f0ba453a7f4318c763911ec13d019"
TABARENA_RAW = (
    "https://raw.githubusercontent.com/autogluon/tabarena/{sha}/packages/tabarena/src/"
    "tabarena/benchmark/task/metadata/{path}"
)
CURATED_URL = TABARENA_RAW.format(
    sha=TABARENA_SHA, path="data/curated_tabarena_dataset_metadata.csv"
)
BEYOND_CSV_URL = TABARENA_RAW.format(
    sha=TABARENA_SHA, path="sources/data/BeyondArena_tasks_metadata.csv"
)
BEYOND_REPO = "TabArena/BeyondArena"
BEYOND_REV = "2ecfe882ccfb814fc27c4de10a64ceefd5d7655c"
BEYOND_FILE = "https://huggingface.co/datasets/" + BEYOND_REPO + "/resolve/{rev}/{path}"
BEYOND_TREE = (
    "https://huggingface.co/api/datasets/" + BEYOND_REPO + "/tree/{rev}?recursive=true&expand=false"
)
LIST_URL = (
    f"{OPENML}/data/list/status/active/number_instances/500..100000/number_features/3..101"
    "/limit/50000"
)
USER_AGENT = "tabsil-real-corpus-check/1.0"
MISSING = "\x00"
SEP = "\x1f"
TEXT_PATTERN = re.compile(
    r"\b(time[ _-]?series|forecast\w*|images?|vision|text|nlp|audio|speech|video)\b"
)
DEFAULT_CELLS = 20_000_000
RANGE_BLOCK = 1 << 20
DEV_SUITE = "dev-amlb-ctr23"


class Missing(Exception):
    """A resource the source does not have (OpenML answers 4xx for it)."""


class Web:
    """Downloads into `cache`: API answers as cached JSON, files once. Requests are sequential,
    with `pause` seconds between them."""

    def __init__(self, cache: Path, pause: float):
        self.cache = cache
        self.pause = pause

    def get(self, url: str, dest: Path) -> Path:
        if dest.exists():
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(request, timeout=180) as response:
                    with open(part, "wb") as out:
                        while chunk := response.read(1 << 20):
                            out.write(chunk)
                break
            except urllib.error.HTTPError as error:
                part.unlink(missing_ok=True)
                if error.code < 500:
                    raise Missing(f"HTTP {error.code}") from error
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
            except OSError:
                part.unlink(missing_ok=True)
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        os.replace(part, dest)
        time.sleep(self.pause)
        return dest

    def json(self, url: str):
        key = hashlib.sha256(url.encode()).hexdigest()[:32]
        return json.loads(self.get(url, self.cache / "api" / f"{key}.json").read_text("utf-8"))


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def normalize_name(name: str) -> str:
    """Lower case; a version suffix ("_v2", "(version 2)") and all separators removed."""
    text = name.lower().strip()
    text = re.sub(r"\(\s*v(?:ersion)?[\s_.-]*\d+\s*\)$", "", text)
    text = re.sub(r"[\s_.\-]*v(?:ersion)?[\s_.\-]*\d+$", "", text)
    return re.sub(r"[^a-z0-9]", "", text)


def normalize_feature(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def shape_close(rows: int, predictors: int, bench_rows: int, bench_predictors: int) -> bool:
    return abs(rows - bench_rows) <= 0.01 * bench_rows and predictors == bench_predictors


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()


def numbers_to_text(values) -> list[str]:
    """Each number with 6 significant digits; missing and infinite values as markers."""
    x = np.asarray(values, dtype="float64")
    out = np.empty(x.shape, dtype=object)
    finite = np.isfinite(x)
    v = x[finite]
    magnitude = np.where(v == 0, 1.0, np.abs(v))
    factor = 10.0 ** (5.0 - np.floor(np.log10(magnitude)))
    rounded = np.round(v * factor) / factor + 0.0  # + 0.0 turns -0.0 into 0.0
    out[finite] = [f"{t:.6g}" for t in rounded]
    out[np.isnan(x)] = MISSING
    out[np.isposinf(x)] = "+inf"
    out[np.isneginf(x)] = "-inf"
    return out.tolist()


def canonical_column(series: pd.Series) -> list[str]:
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return numbers_to_text(series.to_numpy(dtype="float64", na_value=np.nan))
    return [MISSING if pd.isna(v) else str(v).strip() for v in series.tolist()]


def row_keys(frame: pd.DataFrame) -> tuple[list[bytes], list[bytes]]:
    """The (with names, names ignored) hash of every row, in row order."""
    cols = sorted(frame.columns, key=str)
    if not cols or len(frame) == 0:
        return [], []
    texts = [canonical_column(frame[c]) for c in cols]
    named_cells = [[f"{c}={v}" for v in t] for c, t in zip(cols, texts, strict=True)]
    named = [_digest(SEP.join(row)) for row in zip(*named_cells, strict=True)]
    values = [_digest(SEP.join(sorted(row))) for row in zip(*texts, strict=True)]
    return named, values


def build_index(sources: dict[object, set[bytes]]) -> dict[bytes, list[object]]:
    index: dict[bytes, list[object]] = {}
    for key, hashes in sources.items():
        for h in hashes:
            index.setdefault(h, []).append(key)
    return index


def first_over(index: dict[bytes, list[object]], hashes: list[bytes]) -> tuple[object, int] | None:
    """The source with the most rows among `hashes` if it holds more than 1% of them."""
    counts: Counter = Counter()
    for h in hashes:
        counts.update(index.get(h, ()))
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if count > 0.01 * len(hashes):
            return key, count
    return None


def to_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def is_true(value) -> bool:
    return str(value).lower() == "true"


def attribute_roles(features: list[dict]) -> tuple[str, list[dict], int]:
    """(target name, predictor attributes, missing predictor cells)."""
    target = next((f["name"] for f in features if is_true(f.get("is_target"))), "")
    predictors = [
        f
        for f in features
        if f["name"] != target
        and not is_true(f.get("is_ignore"))
        and not is_true(f.get("is_row_identifier"))
    ]
    missing = sum(to_int(f.get("number_of_missing_values")) or 0 for f in predictors)
    return target, predictors, missing


@dataclass
class Bench:
    suite: str
    name: str
    openml_id: int | None = None
    task_id: int | None = None
    rows: int | None = None
    predictors: list[str] | None = None
    target: str = ""
    dev_member: bool = False
    aliases: list[str] = field(default_factory=list)
    source: str = ""
    note: str = ""

    @property
    def label(self) -> str:
        ident = f" (openml {self.openml_id})" if self.openml_id else ""
        return f"{self.suite}:{self.name}{ident}"


@dataclass
class Source:
    key: str
    labels: list[str] = field(default_factory=list)
    rows: int | None = None
    named: set[bytes] | None = None
    values: set[bytes] | None = None
    profiles: list | None = None  # rule 7's column profiles
    note: str = ""

    @property
    def hashed(self) -> bool:
        return self.named is not None or self.values is not None


@dataclass
class Cand:
    id: int
    name: str
    version: str
    rows: int
    classes: int
    target: str
    predictors: list[str]
    tags: list[str]
    desc: dict
    features: list[dict]


def openml_info(web: Web, did: int, quals: dict) -> tuple[dict, list[dict], dict]:
    desc = web.json(f"{OPENML}/data/{did}")["data_set_description"]
    feats = web.json(f"{OPENML}/data/features/{did}")["data_features"]["feature"]
    return desc, feats, quals


def batch_qualities(web: Web, ids: list[int]) -> dict[int, dict]:
    """Qualities of many datasets, 50 ids per list call (the list endpoint takes ids joined
    with commas); an id the list does not return is absent from the result."""
    found: dict[int, dict] = {}
    for start in range(0, len(ids), 50):
        chunk = ids[start : start + 50]
        listed = web.json(f"{OPENML}/data/list/data_id/{','.join(map(str, chunk))}")
        for item in listed["data"]["dataset"]:
            found[int(item["did"])] = {q["name"]: q["value"] for q in item["quality"]}
    return found


def load_openml(web: Web, did: int, desc: dict, features: list[dict]) -> pd.DataFrame:
    """The rows of an OpenML dataset (without ignored attributes): the parquet file of the
    description if its columns hold every attribute, else the ARFF file."""
    keep = [
        f["name"]
        for f in features
        if not (is_true(f.get("is_ignore")) or is_true(f.get("is_row_identifier")))
    ]
    if desc.get("parquet_url"):
        try:
            frame = pd.read_parquet(web.get(desc["parquet_url"], web.cache / "data" / f"{did}.pq"))
            if set(keep) <= set(map(str, frame.columns)):
                return frame[keep]
        except (OSError, ValueError, KeyError):
            pass
    path = web.get(OPENML_FILE.format(file_id=desc["file_id"]), web.cache / "data" / f"{did}.arff")
    with open(path, encoding="utf-8") as stream:
        parsed = arff.load(stream)
    frame = pd.DataFrame(parsed["data"], columns=[a[0] for a in parsed["attributes"]])
    return frame[[n for n in keep if n in frame.columns]]


def write_table(frame: pd.DataFrame, path: Path) -> None:
    out = frame.copy()
    for column in out.columns:
        if not pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].map(lambda v: None if pd.isna(v) else str(v))
    out.to_parquet(path, index=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RangeFile(io.RawIOBase):
    """A read-only file over HTTP range requests, so that a parquet footer can be read without
    downloading the file (each read is one request for the bytes it asks for)."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def _block(self, index: int) -> bytes:
        """The 1 MiB block `index`, fetched once: many small reads share one request."""
        if index not in self.blocks:
            start = index * RANGE_BLOCK
            end = min(self.size, start + RANGE_BLOCK) - 1
            request = urllib.request.Request(
                self.url, headers={"User-Agent": USER_AGENT, "Range": f"bytes={start}-{end}"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != end - start + 1:
                raise OSError(f"range request to {self.url} was not honored")
            self.blocks[index] = data
        return self.blocks[index]

    def readinto(self, buffer) -> int:
        count = min(len(buffer), self.size - self.position)
        out = bytearray()
        while len(out) < count:
            index, offset = divmod(self.position, RANGE_BLOCK)
            block = self._block(index)
            take = min(count - len(out), len(block) - offset)
            out += block[offset : offset + take]
            self.position += take
        buffer[: len(out)] = out
        return len(out)


def hf_schema(url: str, size: int) -> tuple[int, list[str]]:
    """(rows, column names) of a remote parquet file, from its footer alone."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(RangeFile(url, size))
    return parquet.metadata.num_rows, list(parquet.schema_arrow.names)


def beyond_entries(web: Web, sizes: dict[str, int]) -> list[Bench]:
    """BeyondArena's 142 datasets: shape and target from the metadata, OpenML id from the
    dataset's source link, feature names and rows from its parquet file when the repo has one."""
    text = web.get(BEYOND_CSV_URL, web.cache / "beyond" / "tasks_metadata.csv").read_text("utf-8")
    rows: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(text)):
        rows.setdefault(row["dataset_name"], row)
    if len(rows) != 142:
        raise RuntimeError(f"BeyondArena metadata has {len(rows)} datasets, expected 142")
    entries = []
    for name, row in sorted(rows.items()):
        uri = row["data_foundry_uri"]
        bench = Bench(
            "beyondarena", name, rows=to_int(row["num_instances"]), target=row["target_name"]
        )
        path = f"{uri}/dataset.parquet"
        if path in sizes:
            meta = web.json(
                BEYOND_FILE.format(
                    rev=BEYOND_REV, path=f"{uri}/dataset_metadata.dataset-mold-v1.json"
                )
            )
            link = meta.get("original_dataset_source_download_link", "")
            found = re.search(r"openml\.org/(?:d|datasets?)/(\d+)", link)
            if found:
                bench.openml_id = int(found.group(1))
            rows_in_file, names = hf_schema(
                BEYOND_FILE.format(rev=BEYOND_REV, path=path), sizes[path]
            )
            bench.rows = rows_in_file
            bench.predictors = [
                c for c in names if c != bench.target and not c.startswith("__index_level")
            ]
            bench.source = f"hf:{path}"
        else:
            bench.note = "no parquet in TabArena/BeyondArena at this revision"
        entries.append(bench)
    return entries


def collect_benchmarks(web: Web, args) -> tuple[list[Bench], dict[str, Source]]:
    """The benchmark side: every dataset the project evaluates classification or regression on,
    and the row hashes of each source (within the caps)."""
    entries: list[Bench] = []
    curated = web.get(CURATED_URL, web.cache / "tabarena" / "curated.csv")
    for row in csv.DictReader(io.StringIO(curated.read_text("utf-8"))):
        entries.append(
            Bench("tabarena-51", row["dataset_name"], int(row["dataset_id"]), int(row["task_id"]))
        )
    if len(entries) != 51:
        raise RuntimeError(f"TabArena metadata has {len(entries)} datasets, expected 51")
    dev = [*amlb.PAPER_CLASSIFICATION, *amlb.PAPER_REGRESSION]
    if len(dev) != 57:
        raise RuntimeError(f"amlb.py has {len(dev)} tasks, expected 57")
    for name, task_id, did, _ in dev:
        entries.append(
            Bench(DEV_SUITE, name, did, task_id, dev_member=task_id not in amlb.TABARENA_OVERLAP)
        )
    tree = web.json(BEYOND_TREE.format(rev=BEYOND_REV))
    sizes = {x["path"]: x["size"] for x in tree if x.get("type") == "file"}
    entries.extend(beyond_entries(web, sizes))
    for name, npz in adbench.DATASETS.items():
        entries.append(
            Bench("adbench", name, source=f"adbench:{npz}", note="npz has no feature names")
        )
    for name, (did, _, _) in impute.TABLES.items():
        entries.append(Bench("impute", name, openml_id=did))

    info: dict[int, tuple[dict, list[dict], dict] | None] = {}
    quals_by_id = batch_qualities(web, sorted({b.openml_id for b in entries if b.openml_id}))
    for bench in entries:
        if bench.openml_id is None:
            continue
        if bench.openml_id not in info:
            try:
                if bench.openml_id not in quals_by_id:
                    raise Missing("not in the OpenML list (inactive or absent)")
                info[bench.openml_id] = openml_info(
                    web, bench.openml_id, quals_by_id[bench.openml_id]
                )
            except Missing as error:
                info[bench.openml_id] = None
                bench.note = f"OpenML: {error}"
        if info[bench.openml_id] is None:
            continue
        desc, feats, quals = info[bench.openml_id]
        bench.aliases.append(desc["name"])
        target, predictors, _ = attribute_roles(feats)
        bench.target = bench.target or target
        bench.rows = bench.rows or to_int(quals.get("NumberOfInstances"))
        if bench.predictors is None:
            bench.predictors = [p["name"] for p in predictors]
        if not bench.source:
            bench.source = f"openml:{bench.openml_id}"

    sources: dict[str, Source] = {}
    for bench in entries:
        if bench.source:
            sources.setdefault(bench.source, Source(bench.source)).labels.append(bench.label)
    for key in sorted(sources):
        source = sources[key]
        if time.monotonic() > args.deadline:
            source.note = "not hashed: time budget"
            continue
        try:
            hash_source(web, args, key, source, info, sizes)
        except (Missing, ValueError) as error:
            source.note = f"not hashed: {error}"
        log(f"  {key}: rows {source.rows} {source.note}")
    for bench in entries:
        if bench.source in sources and sources[bench.source].rows is not None:
            bench.rows = sources[bench.source].rows
    return entries, sources


def hash_source(web: Web, args, key: str, source: Source, info, sizes) -> None:
    kind, _, ident = key.partition(":")
    if kind == "openml":
        did = int(ident)
        if info.get(did) is None:
            raise Missing("no OpenML description")
        desc, feats, quals = info[did]
        cells = (to_int(quals.get("NumberOfInstances")) or 0) * len(feats)
        if cells > args.max_benchmark_cells:
            source.note = f"not hashed: {cells} cells over the cap"
            return
        frame = load_openml(web, did, desc, feats)
    elif kind == "hf":
        size = sizes[ident]
        if size > args.max_benchmark_mb * 1e6:
            source.note = f"not hashed: parquet of {size / 1e6:.0f} MB over the cap"
            return
        path = web.get(
            BEYOND_FILE.format(rev=BEYOND_REV, path=ident),
            web.cache / "beyond" / ident.replace("/", "__"),
        )
        frame = pd.read_parquet(path)
        frame = frame[[c for c in frame.columns if not str(c).startswith("__index_level")]]
    elif kind == "adbench":
        path = _files.download_verified(
            f"{adbench.ADBENCH_URL}/{ident}.npz",
            web.cache / "adbench" / f"{ident}.npz",
            adbench.SHA256[ident],
            what=ident,
        )
        with np.load(path) as data:
            x, y = data["X"], data["y"]
        frame = pd.DataFrame(np.column_stack([x.astype(float), y.astype(float)]))
        frame.columns = [f"c{i}" for i in range(frame.shape[1])]
        source.rows = len(frame)
        source.values = set(row_keys(frame)[1])
        source.profiles = column_profiles(frame, SAMPLE)
        return
    else:
        raise ValueError(f"unknown source {key}")
    named, values = row_keys(frame)
    source.rows = len(frame)
    source.named = set(named)
    source.values = set(values)
    source.profiles = column_profiles(frame, SAMPLE)


def filter_reason(desc, target, features, predictors, missing, rows, classes) -> str | None:
    if desc.get("status") != "active":
        return "not active"
    if not desc.get("default_target_attribute") or not target:
        return "no default target attribute"
    target_type = next(f["data_type"] for f in features if f["name"] == target)
    if classes == 0 and target_type != "numeric":
        return f"regression with a {target_type} target"
    if classes > 0 and target_type != "nominal":
        return f"classification with a {target_type} target"
    if not 2 <= len(predictors) <= 100:
        return f"{len(predictors)} predictors (need 2 to 100)"
    if predictors and missing / (rows * len(predictors)) > 0.5:
        return "more than 50% missing predictor cells"
    tags = desc.get("tag", [])
    tags = [tags] if isinstance(tags, str) else list(tags)
    text = " ".join([str(desc.get("name", "")), *map(str, tags)]).lower()
    match = TEXT_PATTERN.search(text)
    if match:
        return f"time-series, image or text by name or tag ('{match.group(0)}')"
    match = SERIES_OR_IMAGE_FAMILY.search(str(desc.get("name", "")))
    if match:
        return f"price series or image features by name family ('{match.group(0)}')"
    return None


SYNTHETIC_NAME = re.compile(
    r"^(fri_c\d|fried|2dplanes$|mv$|bank(8|32)|puma(8|32)|kin(8|32)|bng\(|led(\d|_|-|$)"
    r"|waveform|gametes|parity\d|threeof9|mabbob|monks|xd6|corral|mux6|jungle_chess"
    r"|dummy$|mytestdataset|test_dataset|fourclass|comet_mc|epistasis|delta_(elevators|ailerons)"
    r"|elevators$|ailerons$)",
    re.IGNORECASE,
)
# FOREX price series (up or down per day or hour) and Meta-Album image features
SERIES_OR_IMAGE_FAMILY = re.compile(r"^(forex|meta_album)", re.IGNORECASE)
SYNTHETIC_TEXT = re.compile(
    r"\b(synthetic\w*|artificial\w*|simulat\w*|generated|generators?)\b", re.IGNORECASE
)


def synthetic_reason(desc: dict) -> str | None:
    """Step 'synthetic': a table drawn from a known generator, by its name family (Friedman,
    formula or Delve simulator families) or by its OpenML description. Over-exclusion is
    accepted; a synthetic table in the corpus is not."""
    name = str(desc.get("name", ""))
    if SYNTHETIC_NAME.search(name):
        return f"generator name family ('{name}')"
    match = SYNTHETIC_TEXT.search(str(desc.get("description", "")))
    if match:
        return f"OpenML description says '{match.group(0).lower()}'"
    return None


def name_stem(name: str) -> str:
    """The first token of a dataset name ('disclosure' of 'disclosure_x_noise')."""
    return re.split(r"[_\-\s]", name.strip().lower())[0]


# Rule 7 (module docstring): columns compared without names, category labels or row order
LOW_CARD = 64  # 2 to this many distinct values: compared by the category shares
MIN_DISTINCT = 30  # more distinct values than this: compared by the values
MIN_DISTINCT_INT = 100  # ... for a column of integer values
SAMPLE = 1000  # value hashes kept per benchmark column, the smallest (consistent sampling)
MIN_SAMPLED = 8  # sampled values a containment estimate needs
CONTAINED = 0.8  # share of a column's sampled values a benchmark column must hold
SHARES_L1 = 0.05  # largest L1 distance of two columns' sorted category shares (a subset's noise)
ALL_HASHES = int(np.iinfo(np.uint64).max)


@dataclass
class ColumnProfile:
    """Rule 7's view of one column: the sorted category shares of a low-cardinality column, or
    the sorted hashes of the distinct values of another (as they are, and min-max normalized for
    numbers). A benchmark profile keeps the `SAMPLE` smallest hashes and the largest kept one
    (`raw_cut`, `norm_cut`); a candidate profile keeps them all."""

    shares: np.ndarray | None = None
    raw: np.ndarray | None = None
    raw_cut: int = ALL_HASHES
    norm: np.ndarray | None = None
    norm_cut: int = ALL_HASHES


def value_hashes(texts) -> np.ndarray:
    """The sorted distinct 64-bit hashes of `texts`."""
    joined = b"".join(_digest(t) for t in texts)
    return np.unique(np.frombuffer(joined, dtype=np.uint64))


def _sampled(hashes: np.ndarray, sample: int | None) -> tuple[np.ndarray, int]:
    if sample is None or len(hashes) <= sample:
        return hashes, ALL_HASHES
    return hashes[:sample], int(hashes[sample - 1])


def column_profiles(frame: pd.DataFrame, sample: int | None = None) -> list[ColumnProfile]:
    """The profile of every column rule 7 compares (constant columns and value columns with too
    few distinct values are left out). `sample`: keep that many value hashes per column."""
    profiles = []
    for column in frame.columns:
        series = frame[column]
        text = canonical_column(series)
        counts = Counter(t for t in text if t != MISSING)
        if len(counts) < 2:
            continue
        if len(counts) <= LOW_CARD:
            shares = np.sort(np.fromiter(counts.values(), dtype=np.float64))[::-1]
            profiles.append(ColumnProfile(shares=shares / shares.sum()))
            continue
        numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        x = None
        if numeric:
            x = series.to_numpy(dtype="float64", na_value=np.nan)
            x = x[np.isfinite(x)]
            integer = bool(np.all(x == np.round(x)))
            if len(counts) <= (MIN_DISTINCT_INT if integer else MIN_DISTINCT):
                continue
        elif len(counts) <= MIN_DISTINCT:
            continue
        raw, raw_cut = _sampled(value_hashes(counts), sample)
        profile = ColumnProfile(raw=raw, raw_cut=raw_cut)
        if x is not None and x.max() > x.min():
            normalized = set(numbers_to_text((x - x.min()) / (x.max() - x.min())))
            profile.norm, profile.norm_cut = _sampled(value_hashes(normalized), sample)
        profiles.append(profile)
    return profiles


class ColumnIndex:
    """Rule 7's lookup over the benchmark sources' column profiles."""

    def __init__(self, sources: dict[str, Source]) -> None:
        self.keys = sorted(k for k, s in sources.items() if s.profiles)
        categories: dict[int, list[tuple[np.ndarray, int]]] = {}
        values: list[tuple[int, ColumnProfile]] = []
        for i, key in enumerate(self.keys):
            for profile in sources[key].profiles:
                if profile.shares is not None:
                    categories.setdefault(len(profile.shares), []).append((profile.shares, i))
                else:
                    values.append((i, profile))
        self.categories = {
            n: (np.stack([s for s, _ in rows]), np.array([i for _, i in rows]))
            for n, rows in categories.items()
        }
        self.value_source = np.array([i for i, _ in values], dtype=np.int64)
        self.raw = self._lookup([(j, p.raw, p.raw_cut) for j, (_, p) in enumerate(values)])
        self.norm = self._lookup(
            [(j, p.norm, p.norm_cut) for j, (_, p) in enumerate(values) if p.norm is not None]
        )

    def _lookup(self, columns: list[tuple[int, np.ndarray, int]]):
        """(sorted hashes, their value column, each value column's cut; 0 for a column that is
        not in `columns`, so that no candidate hash is sampled for it)."""
        cuts = np.zeros(len(self.value_source), dtype=np.uint64)
        if not columns:
            return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64), cuts
        hashes = np.concatenate([h for _, h, _ in columns])
        owner = np.repeat([j for j, _, _ in columns], [len(h) for _, h, _ in columns])
        for j, _, cut in columns:
            cuts[j] = cut
        order = np.argsort(hashes, kind="stable")
        return hashes[order], owner[order], cuts

    @staticmethod
    def _holding(lookup, hashes: np.ndarray) -> np.ndarray:
        """The value columns that hold at least `CONTAINED` of the sampled `hashes`."""
        sorted_hashes, owner, cuts = lookup
        left = np.searchsorted(sorted_hashes, hashes, "left")
        lengths = np.searchsorted(sorted_hashes, hashes, "right") - left
        starts, lengths = left[lengths > 0], lengths[lengths > 0]
        counts = np.zeros(len(cuts), dtype=np.int64)
        if len(starts):
            offsets = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
            counts = np.bincount(owner[offsets + np.arange(lengths.sum())], minlength=len(cuts))
        sampled = np.searchsorted(hashes, cuts, "right")
        return np.flatnonzero((sampled >= MIN_SAMPLED) & (counts >= CONTAINED * sampled))

    def match(self, profiles: list[ColumnProfile]) -> tuple[str, str] | None:
        """(benchmark source, reason) of the first source rule 7 matches, or None."""
        compared = 0
        value_hits: Counter = Counter()
        category_hits: Counter = Counter()
        for profile in profiles:
            if profile.shares is not None:
                if len(profile.shares) < 3:
                    continue  # binary columns match too easily to count
                compared += 1
                block = self.categories.get(len(profile.shares))
                if block is not None:
                    shares, owner = block
                    close = np.abs(shares - profile.shares).sum(axis=1) <= SHARES_L1
                    category_hits.update(set(owner[close].tolist()))
                continue
            compared += 1
            columns = set(self._holding(self.raw, profile.raw).tolist())
            if profile.norm is not None:
                columns |= set(self._holding(self.norm, profile.norm).tolist())
            value_hits.update({int(self.value_source[j]) for j in columns})
        ranked = sorted(
            set(value_hits) | set(category_hits), key=lambda i: -value_hits[i] - category_hits[i]
        )
        for i in ranked:
            matched = value_hits[i] + category_hits[i]
            if value_hits[i] >= 2 or (matched >= 3 and matched >= 0.6 * compared):
                return self.keys[i], (
                    f"7: {value_hits[i]} value and {category_hits[i]} category columns of"
                    f" {compared} compared match one benchmark dataset's columns"
                )
        return None


def candidate_pool(web: Web, args) -> tuple[list[Cand], list[dict], int, int]:
    """Candidates in id order that pass the filters (up to `args.max_candidates`), the filter
    exclusions, the number examined and the number left unexamined past the cap."""
    listed = web.json(LIST_URL)["data"]["dataset"]
    pre = []
    for item in listed:
        quals = {q["name"]: q["value"] for q in item["quality"]}
        classes = to_int(quals.get("NumberOfClasses"))
        if classes is None or not (classes == 0 or 2 <= classes <= 10):
            continue
        pre.append((int(item["did"]), item["name"], quals, classes))
    pre.sort(key=lambda row: row[0])
    cands: list[Cand] = []
    excluded: list[dict] = []
    examined = 0
    for did, name, quals, classes in pre:
        if len(cands) >= args.max_candidates or time.monotonic() > args.pool_deadline:
            break
        examined += 1
        rows = to_int(quals.get("NumberOfInstances")) or 0
        try:
            desc = web.json(f"{OPENML}/data/{did}")["data_set_description"]
            features = web.json(f"{OPENML}/data/features/{did}")["data_features"]["feature"]
        except Missing as error:
            excluded.append(exclusion(did, name, "filter", f"no description or features ({error})"))
            continue
        except OSError as error:  # a server error that outlasted the retries (HTTP 5xx, timeout)
            reason = f"description or features not served ({type(error).__name__}: {error})"
            excluded.append(exclusion(did, name, "filter", reason))
            continue
        target, predictors, missing = attribute_roles(features)
        reason = filter_reason(desc, target, features, predictors, missing, rows, classes)
        if reason:
            excluded.append(exclusion(did, name, "filter", reason))
            continue
        generated = synthetic_reason(desc)
        if generated:
            excluded.append(exclusion(did, name, "synthetic", generated))
            continue
        tags = desc.get("tag", [])
        cands.append(
            Cand(
                id=did,
                name=desc["name"],
                version=str(desc.get("version", "")),
                rows=rows,
                classes=classes,
                target=target,
                predictors=[p["name"] for p in predictors],
                tags=[tags] if isinstance(tags, str) else list(tags),
                desc=desc,
                features=features,
            )
        )
    return cands, excluded, examined, len(pre) - examined


def exclusion(did: int, name: str, step: str, reason: str, matched: str = "") -> dict:
    return {"openml_id": did, "name": name, "step": step, "reason": reason, "matched": matched}


def metadata_match(cand: Cand, bench: list[Bench], by_id: dict, by_name: dict):
    """Rules 1 to 3, on the candidate's id, name, shape and predictor names. Returns
    (rule, matched benchmark label) or None."""
    if cand.id in by_id:
        return "1: same OpenML dataset id", by_id[cand.id].label
    hit = by_name.get(normalize_name(cand.name))
    if hit is not None:
        return "1: name equals a benchmark name", hit.label
    names = {normalize_feature(p) for p in cand.predictors}
    compared = [b for b in bench if b.predictors is not None and b.rows is not None]
    for b in compared:
        other = {normalize_feature(p) for p in b.predictors}
        if shape_close(cand.rows, len(cand.predictors), b.rows, len(b.predictors)) and (
            jaccard(names, other) >= 0.5
        ):
            return "2: same shape and predictor-name Jaccard >= 0.5", b.label
    for b in compared:
        if jaccard(names, {normalize_feature(p) for p in b.predictors}) >= 0.8:
            return "3: predictor-name Jaccard >= 0.8", b.label
    for b in bench:
        if b.rows and abs(cand.rows - b.rows) <= 0.01 * b.rows:
            return "6: row count within 1% of a benchmark dataset's", b.label
    return None


def run(args) -> None:
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    web = Web(Path(args.cache), args.pause)
    args.deadline = time.monotonic() + args.max_minutes * 60
    t0 = time.monotonic()

    log("benchmark side")
    bench, sources = collect_benchmarks(web, args)
    by_id = {b.openml_id: b for b in bench if b.openml_id is not None}
    by_name: dict[str, Bench] = {}
    for b in bench:
        for name in [b.name, *b.aliases]:
            by_name.setdefault(normalize_name(name), b)
    named_index = build_index({k: s.named for k, s in sources.items() if s.named})
    value_index = build_index({k: s.values for k, s in sources.items() if s.values})
    column_index = ColumnIndex(sources)
    log(
        f"benchmark datasets {len(bench)}; sources hashed {sum(s.hashed for s in sources.values())}"
        f" of {len(sources)}"
    )

    log("candidates")
    args.pool_deadline = time.monotonic() + args.max_pool_minutes * 60
    cands, excluded, examined, not_examined = candidate_pool(web, args)
    log(f"passing the filters {len(cands)} of {examined} examined")
    status: dict[int, str] = {c.id: "" for c in cands}
    rows_out = list(excluded)
    not_checked: list[str] = []

    survivors = []
    for cand in cands:
        hit = metadata_match(cand, bench, by_id, by_name)
        if hit:
            rule, matched = hit
            rows_out.append(exclusion(cand.id, cand.name, rule[0], rule, matched))
            status[cand.id] = f"excluded {rule[0]}"
        else:
            survivors.append(cand)
    # variants of one source: the same name stem and row count, the lowest id kept (step 5)
    first_of_variant: dict[tuple[str, int], int] = {}
    unique: list[Cand] = []
    for cand in survivors:
        key = (name_stem(cand.name), cand.rows)
        if key in first_of_variant:
            reason = "5: variant of a lower-id candidate (same name stem and row count)"
            rows_out.append(exclusion(cand.id, cand.name, "5", reason, str(first_of_variant[key])))
            status[cand.id] = "excluded 5"
        else:
            first_of_variant[key] = cand.id
            unique.append(cand)
    survivors = unique
    log(f"after rules 1 to 3 and the variant dedupe: {len(survivors)}")

    kept: list[Cand] = []
    stats: dict[int, dict] = {}
    seen_names: dict[str, int] = {}
    kept_named: dict[bytes, list[object]] = {}
    kept_values: dict[bytes, list[object]] = {}
    for cand in survivors:
        if time.monotonic() > args.deadline:
            status[cand.id] = "not checked: time budget"
            not_checked.append(f"{cand.id} {cand.name}: time budget")
            continue
        try:
            frame = load_openml(web, cand.id, cand.desc, cand.features)
        except Exception as error:  # a failed load is recorded, not imputed
            status[cand.id] = "not checked: load failed"
            not_checked.append(f"{cand.id} {cand.name}: load failed ({type(error).__name__})")
            continue
        predictors = [c for c in cand.predictors if c in frame.columns]
        cells = len(frame) * len(predictors)
        missing = float(frame[predictors].isna().sum().sum()) / cells if cells else 0.0
        classes = int(frame[cand.target].nunique(dropna=True)) if cand.target in frame else 0
        post = None
        if cand.target not in frame.columns:
            post = "target missing after download"
        elif cand.classes > 0 and not 2 <= classes <= 10:
            post = f"{classes} classes after download"
        elif missing > 0.5:
            post = "more than 50% missing after download"
        if post:
            rows_out.append(exclusion(cand.id, cand.name, "filter", post))
            status[cand.id] = "excluded filter"
            continue
        named, values = row_keys(frame)
        n = len(named)
        leak = first_over(named_index, named)
        kind = "names kept"
        if leak is None:
            leak = first_over(value_index, values)
            kind = "names ignored"
        if leak is not None:
            key, count = leak
            label = "; ".join(sources[key].labels[:3])
            rule = f"4: row hash ({kind}) shared with a benchmark for {count} of {n} rows"
            rows_out.append(exclusion(cand.id, cand.name, "4", rule, label))
            status[cand.id] = "excluded 4"
            continue
        shared = column_index.match(column_profiles(frame[[*predictors, cand.target]]))
        if shared is not None:
            key, rule = shared
            label = "; ".join(sources[key].labels[:3])
            rows_out.append(exclusion(cand.id, cand.name, "7", rule, label))
            status[cand.id] = "excluded 7"
            continue
        norm = normalize_name(cand.name)
        if norm in seen_names:
            rows_out.append(
                exclusion(
                    cand.id,
                    cand.name,
                    "5",
                    "5: same normalized name as a lower-id candidate",
                    str(seen_names[norm]),
                )
            )
            status[cand.id] = "excluded 5"
            continue
        dup = first_over(kept_named, named) or first_over(kept_values, values)
        if dup is not None:
            key, count = dup
            rows_out.append(
                exclusion(
                    cand.id,
                    cand.name,
                    "5",
                    f"5: row hash shared with a lower-id candidate for {count} of {n} rows",
                    str(key),
                )
            )
            status[cand.id] = "excluded 5"
            continue
        seen_names[norm] = cand.id
        for h in set(named):
            kept_named.setdefault(h, []).append(cand.id)
        for h in set(values):
            kept_values.setdefault(h, []).append(cand.id)
        table = out / "tables" / f"{cand.id}.parquet"
        write_table(frame[[*predictors, cand.target]], table)
        task = "regression" if cand.classes == 0 else ("binary" if classes == 2 else "multiclass")
        stats[cand.id] = {
            "rows": len(frame),
            "predictors": len(predictors),
            "missing": missing,
            "categorical": sum(not pd.api.types.is_numeric_dtype(frame[c]) for c in predictors),
            "classes": classes if cand.classes > 0 else "",
            "task": task,
            "sha": sha256_file(table),
            "bytes": table.stat().st_size,
        }
        status[cand.id] = "kept"
        kept.append(cand)
        log(f"kept {cand.id} {cand.name} ({task}, {len(frame)} rows)")

    write_outputs(out, cands, status, rows_out, bench, sources, kept, stats)
    write_summary(
        out,
        args,
        examined,
        not_examined,
        cands,
        rows_out,
        bench,
        sources,
        kept,
        stats,
        not_checked,
        time.monotonic() - t0,
    )


def write_outputs(out, cands, status, rows_out, bench, sources, kept, stats) -> None:
    by_id = {c.id: c for c in cands}
    pd.DataFrame(
        [
            {
                "openml_id": c.id,
                "name": c.name,
                "version": c.version,
                "rows": c.rows,
                "features": stats[c.id]["predictors"],
                "categorical_features": stats[c.id]["categorical"],
                "task_type": stats[c.id]["task"],
                "target": c.target,
                "classes": stats[c.id]["classes"],
                "missing_share": round(stats[c.id]["missing"], 6),
                "sha256": stats[c.id]["sha"],
                "bytes": stats[c.id]["bytes"],
                "file": f"tables/{c.id}.parquet",
            }
            for c in kept
        ]
    ).to_csv(out / "manifest.csv", index=False)
    pd.DataFrame(rows_out, columns=["openml_id", "name", "step", "reason", "matched"]).to_csv(
        out / "excluded.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "openml_id": c.id,
                "name": c.name,
                "status": status[c.id],
                "rows": c.rows,
                "predictors": len(c.predictors),
                "classes": c.classes,
            }
            for c in by_id.values()
        ]
    ).to_csv(out / "candidates.csv", index=False)
    pd.DataFrame(
        [
            {
                "suite": b.suite,
                "name": b.name,
                "openml_id": b.openml_id,
                "task_id": b.task_id,
                "rows": b.rows,
                "predictors": None if b.predictors is None else len(b.predictors),
                "feature_names": b.predictors is not None,
                "dev_suite_member": b.dev_member if b.suite == DEV_SUITE else "",
                "source": b.source,
                "row_hashes": bool(b.source in sources and sources[b.source].hashed),
                "note": b.note or (sources[b.source].note if b.source in sources else ""),
            }
            for b in bench
        ]
    ).to_csv(out / "benchmarks.csv", index=False)


def write_summary(
    out,
    args,
    examined,
    not_examined,
    cands,
    rows_out,
    bench,
    sources,
    kept,
    stats,
    not_checked,
    seconds,
) -> None:
    by_step = Counter(row["step"] for row in rows_out)
    by_filter = Counter(row["reason"] for row in rows_out if row["step"] == "filter")
    tasks = Counter(stats[c.id]["task"] for c in kept)
    size = sum(stats[c.id]["bytes"] for c in kept)
    suites = Counter(b.suite for b in bench)
    hashed = Counter(b.suite for b in bench if b.source in sources and sources[b.source].hashed)
    no_hash = []
    for b in bench:
        if b.source in sources and sources[b.source].hashed:
            continue
        reason = b.note or (sources[b.source].note if b.source in sources else "")
        no_hash.append(f"- {b.label}: {reason or 'no data source'}")
    lines = [
        "# Real-table corpus: summary",
        "",
        f"Run time {seconds / 60:.1f} min. Cache `{args.cache}`, output `{args.out}`.",
        "",
        "## Benchmark side",
        "",
        "| suite | datasets | with row hashes |",
        "|---|---|---|",
        *[f"| {s} | {suites[s]} | {hashed.get(s, 0)} |" for s in suites],
        "",
        f"Sources hashed: {sum(s.hashed for s in sources.values())} of {len(sources)}.",
        "",
        "## Candidates",
        "",
        f"Examined {examined} (cap {args.max_candidates}, pool time budget "
        f"{args.max_pool_minutes:g} min); {not_examined} passing-candidate ids were not examined "
        f"(past the cap or the budget). Passing the filters: {len(cands)}.",
        "",
        "Exclusions by step:",
        "",
        *[f"- step {k}: {v}" for k, v in sorted(by_step.items())],
        "",
        "Filter exclusions by reason:",
        "",
        *[f"- {k}: {v}" for k, v in by_filter.most_common()],
        "",
        f"Kept: {len(kept)} ({', '.join(f'{k} {v}' for k, v in sorted(tasks.items()))}), "
        f"{size / 1e6:.1f} MB as parquet.",
        "",
        "## Could not be checked",
        "",
        *([f"- {line}" for line in not_checked] or ["- no candidate was left unchecked"]),
        "",
        "Benchmark datasets without row hashes:",
        "",
        *(no_hash or ["- none"]),
        "",
        "Feature names are predictor names only (the target is not compared). ADBench npz files "
        "have no feature names: rules 2 and 3 skip them and rule 4 uses the names-ignored hash.",
        "",
        "## Rerun",
        "",
        "    uv run --frozen --package tabsil-train python -I training/scripts/real_corpus.py \\",
        f"      --cache {args.cache} --out {args.out} --max-candidates {args.max_candidates}",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-table corpus and leakage check.")
    parser.add_argument("--cache", default="/tmp/top5-corpus-data")
    parser.add_argument("--out", default="/tmp/top5-corpus-out")
    parser.add_argument("--max-candidates", type=int, default=400)
    parser.add_argument("--max-benchmark-cells", type=int, default=DEFAULT_CELLS)
    parser.add_argument("--max-benchmark-mb", type=float, default=300.0)
    parser.add_argument("--max-minutes", type=float, default=55.0)
    parser.add_argument("--max-pool-minutes", type=float, default=25.0)
    parser.add_argument("--pause", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
