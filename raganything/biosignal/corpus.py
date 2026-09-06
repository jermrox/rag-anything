"""A curated corpus of health-signal and BLE wearable code, for RAG ingestion.

The point of this module is to stop reinventing physiology. There is a large
body of validated, open-source work on PPG and HRV analysis -- peak detection,
artifact correction, signal quality indices, pulse-wave morphology -- and a
smaller body of open BLE wearable firmware. Very little of it is in any one
place, and none of it is queryable alongside your own device's data.

So: fetch it, extract its structure, and index it into the knowledge graph next
to the biosignal sessions. Then "how do other people detect a bad PPG beat?" and
"what did my band record last night?" become questions you can ask in the same
place, of the same system.

What gets indexed is a **structural summary**, not raw source: module
docstrings, class and function signatures with their docstrings, and file
inventories. That is deliberate. Raw code chunks badly -- a 1200-token window
lands mid-function and retrieves noise -- while a signature plus its docstring
is a self-contained, semantically complete unit that says what something does
and how to call it.

Licences are recorded per source because this corpus exists to be *built on*,
and a GPL dependency is a product decision, not a footnote.
"""

from __future__ import annotations

import ast
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CORPUS",
    "Source",
    "SourceKindLiteral",
    "extract_python_structure",
    "fetch_source",
    "ingest_corpus",
    "sources_for",
    "to_content_list",
]

SourceKindLiteral = str  # "library" | "firmware" | "model" | "spec" | "reference"


@dataclass(frozen=True)
class Source:
    """One external project worth learning from.

    Attributes:
        name: Short identifier, used as the document id in the graph.
        url: Git URL to clone, or a documentation URL for reference-only entries.
        kind: What sort of thing it is -- library, firmware, model, spec.
        license: SPDX-ish identifier. Recorded because it constrains what you
            can build on top, and that is a decision, not a detail.
        relevance: Why this is in the corpus at all, in one sentence.
        provides: The specific capabilities worth borrowing.
        include: Subpaths to walk. Empty means the whole repository.
        exclude: Directory names to skip anywhere in the tree.
        language: Primary language, which decides how the structure is read.
    """

    name: str
    url: str
    kind: SourceKindLiteral
    license: str
    relevance: str
    provides: Tuple[str, ...] = ()
    include: Tuple[str, ...] = ()
    exclude: Tuple[str, ...] = ("test", "tests", "docs", "examples", "node_modules")
    language: str = "python"

    @property
    def is_clonable(self) -> bool:
        return self.url.endswith(".git") or "github.com" in self.url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "kind": self.kind,
            "license": self.license,
            "relevance": self.relevance,
            "provides": list(self.provides),
            "language": self.language,
        }


#: The curated corpus. Every entry was checked to exist rather than recalled.
#:
#: Ordered roughly by how directly it bears on building an honest PPG band:
#: signal quality first, because knowing when a reading is bad is the whole
#: differentiator; then morphology, because sampling rate is the lever a
#: firmware owner controls; then general physiology toolkits; then firmware.
CORPUS: Tuple[Source, ...] = (
    Source(
        name="vital_sqi",
        url="https://github.com/Oucru-Innovations/vital-sqi",
        kind="library",
        license="MIT",
        relevance=(
            "74 signal quality indices for PPG and ECG -- the published state of "
            "the art in deciding whether a physiological reading is trustworthy, "
            "which is exactly the judgement this project refuses to skip."
        ),
        provides=(
            "signal quality indices (SQI) for PPG and ECG",
            "rules-based classification of usable vs unusable segments",
            "HRV computation inherited from HeartPy and py-ecg-detectors",
        ),
    ),
    Source(
        name="pyPPG",
        url="https://github.com/godamartonaron/GODA_pyPPG",
        kind="library",
        license="MIT",
        relevance=(
            "Extracts 74 morphological PPG biomarkers from fiducial points. This "
            "is the feature class consumer bands discard by sampling too slowly, "
            "and the reason controlling your own firmware matters."
        ),
        provides=(
            "pulse wave segmentation and fiducial point detection",
            "74 morphological biomarkers (augmentation index, reflection index)",
            "derivative-based analysis of the PPG waveform",
            "validated against public PPG databases",
        ),
    ),
    Source(
        name="NeuroKit2",
        url="https://github.com/neuropsychology/NeuroKit",
        kind="library",
        license="MIT",
        relevance=(
            "The broadest validated physiological signal toolkit in Python, "
            "covering ECG, PPG, EDA, EMG and respiration, and benchmarked against "
            "BioSPPy, HeartPy and systole."
        ),
        provides=(
            "PPG and ECG peak detection with multiple published algorithms",
            "HRV time, frequency and non-linear domain metrics",
            "respiration extraction from cardiac signals (EDR/RIIV family)",
            "artifact correction and signal simulation for testing",
        ),
    ),
    Source(
        name="HeartPy",
        url="https://github.com/paulvangentcom/heartrate_analysis_python",
        kind="library",
        license="MIT",
        relevance=(
            "Built specifically for noisy PPG from cheap sensors rather than for "
            "clean clinical ECG, which is the regime a wrist band actually "
            "operates in."
        ),
        provides=(
            "peak detection robust to noisy wearable PPG",
            "time-domain HRV measures",
            "outlier and artifact rejection tuned for consumer signals",
        ),
    ),
    Source(
        name="BioSPPy",
        url="https://github.com/scientisst/BioSPPy",
        kind="library",
        license="BSD-3-Clause",
        relevance=(
            "A long-standing reference implementation of biosignal processing "
            "primitives, useful as a cross-check on any algorithm you write."
        ),
        provides=(
            "filtering and preprocessing primitives",
            "ECG, PPG, EDA, EMG, respiration pipelines",
            "clustering and biometrics utilities",
        ),
    ),
    Source(
        name="systole",
        url="https://github.com/embodied-computation-group/systole",
        kind="library",
        license="GPL-3.0",
        relevance=(
            "Cardiac signal analysis with strong artifact-correction and "
            "interactive inspection. Note the licence: GPL-3.0 is a product "
            "decision, not a footnote."
        ),
        provides=(
            "RR interval artifact detection and correction",
            "HRV metrics with visual diagnostics",
            "instantaneous heart rate estimation",
        ),
    ),
    Source(
        name="pypg",
        url="https://github.com/hpi-dhc/pypg",
        kind="library",
        license="MIT",
        relevance=(
            "A second, independent PPG processing implementation -- valuable "
            "precisely because agreement between independent implementations is "
            "evidence and disagreement is a finding."
        ),
        provides=(
            "PPG preprocessing and segmentation",
            "feature extraction for downstream models",
        ),
    ),
    Source(
        name="zephyr-peripheral-hr",
        url="https://github.com/zephyrproject-rtos/zephyr",
        kind="firmware",
        license="Apache-2.0",
        relevance=(
            "The canonical open implementation of the Heart Rate GATT service on "
            "the peripheral side -- the reference for what a band should "
            "broadcast and how."
        ),
        provides=(
            "Heart Rate Service (0x180D) peripheral implementation",
            "GATT service and characteristic registration patterns",
            "BLE advertising and connection handling on nRF hardware",
        ),
        include=("samples/bluetooth/peripheral_hr", "subsys/bluetooth/services"),
        language="c",
    ),
    Source(
        name="TEGSense",
        url="https://github.com/TEGSense/firmware",
        kind="firmware",
        license="unknown",
        relevance=(
            "A complete battery-free PPG wearable: MAX30101 optical sensor over "
            "BLE on an nRF52832, built on Zephyr. The closest open analogue to "
            "the device being built here."
        ),
        provides=(
            "MAX30101 PPG sensor driver integration",
            "power-constrained sampling strategy",
            "BLE transport for optical sensor data",
        ),
        language="c",
    ),
    Source(
        name="BLE-Watch",
        url="https://github.com/CharlesDias/BLE-Watch",
        kind="firmware",
        license="Apache-2.0",
        relevance=(
            "Shows how to combine adopted BLE services with custom ones on the "
            "same peripheral -- the pattern needed to broadcast standard heart "
            "rate alongside signal-quality fields that no standard service has."
        ),
        provides=(
            "mixing standard and custom GATT services",
            "Zephyr peripheral application structure",
        ),
        language="c",
    ),
    Source(
        name="PaPaGei",
        url="https://github.com/Nokia-Bell-Labs/papagei-foundation-model",
        kind="model",
        license="see repository",
        relevance=(
            "An open foundation model for PPG signals. If the device produces raw "
            "waveform data, this is a pretrained starting point rather than "
            "training from zero."
        ),
        provides=(
            "pretrained PPG representation model",
            "embeddings for downstream physiological tasks",
        ),
    ),
)


def sources_for(
    kind: Optional[str] = None, language: Optional[str] = None
) -> List[Source]:
    """Filter the corpus by kind and/or language."""
    return [
        s
        for s in CORPUS
        if (kind is None or s.kind == kind)
        and (language is None or s.language == language)
    ]


# --------------------------------------------------------------------------
# fetching
# --------------------------------------------------------------------------


def fetch_source(
    source: Source, dest_root: Path, depth: int = 1, timeout: int = 300
) -> Optional[Path]:
    """Shallow-clone one source. Returns the checkout path, or ``None``.

    A failure here is logged and returns ``None`` rather than raising: one
    unreachable repository must not abort ingestion of the other ten.
    """
    if not source.is_clonable:
        logger.info(
            "%s is reference-only (%s); nothing to clone", source.name, source.url
        )
        return None

    dest = Path(dest_root) / source.name
    if dest.exists():
        logger.info("%s already fetched at %s", source.name, dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", str(depth), "--quiet", source.url, str(dest)]
    try:
        subprocess.run(cmd, check=True, timeout=timeout, capture_output=True)
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "could not clone %s (%s): %s",
            source.name,
            source.url,
            (exc.stderr or b"").decode("utf-8", "replace").strip()[:200],
        )
        shutil.rmtree(dest, ignore_errors=True)
        return None
    except subprocess.TimeoutExpired:
        logger.warning("cloning %s timed out after %ss", source.name, timeout)
        shutil.rmtree(dest, ignore_errors=True)
        return None
    return dest


# --------------------------------------------------------------------------
# structure extraction
# --------------------------------------------------------------------------


def _signature(node: ast.AST) -> str:
    """Render a def/class header without its body."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = []
        a = node.args
        for arg in list(a.posonlyargs) + list(a.args):
            args.append(arg.arg)
        if a.vararg:
            args.append("*" + a.vararg.arg)
        for arg in a.kwonlyargs:
            args.append(arg.arg)
        if a.kwarg:
            args.append("**" + a.kwarg.arg)
        prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {node.name}({', '.join(args)})"
    if isinstance(node, ast.ClassDef):
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    return ""


def _first_line(text: Optional[str], limit: int = 300) -> str:
    if not text:
        return ""
    stripped = " ".join(text.strip().split())
    return stripped[:limit]


def extract_python_structure(path: Path) -> Optional[Dict[str, Any]]:
    """Module docstring plus public API surface, from one Python file.

    Returns ``None`` for files that do not parse -- a syntax error in someone
    else's repository is not a reason to fail.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(text)
    except (SyntaxError, ValueError, OSError) as exc:
        logger.debug("skipping unparseable %s: %s", path, exc)
        return None

    entries: List[Dict[str, str]] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name.startswith("_"):
                continue
            entry = {
                "signature": _signature(node),
                "doc": _first_line(ast.get_docstring(node)),
            }
            if isinstance(node, ast.ClassDef):
                methods = [
                    _signature(child)
                    for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not child.name.startswith("_")
                ]
                if methods:
                    entry["methods"] = "; ".join(methods[:20])
            entries.append(entry)

    if not entries and not ast.get_docstring(tree):
        return None
    return {
        "path": str(path),
        "module_doc": _first_line(ast.get_docstring(tree), limit=800),
        "api": entries,
        "n_lines": text.count("\n") + 1,
    }


def _extract_c_header(path: Path, max_chars: int = 1200) -> Optional[Dict[str, Any]]:
    """Leading comment block and function-ish lines from a C/C++ file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    head = text[:max_chars]
    signatures = [
        line.strip()
        for line in text.splitlines()
        if line.strip().endswith(")")
        and "(" in line
        and not line.strip().startswith(("//", "*", "#"))
    ][:40]
    if not head.strip() and not signatures:
        return None
    return {
        "path": str(path),
        "module_doc": _first_line(head, limit=800),
        "signatures": signatures,
        "n_lines": text.count("\n") + 1,
    }


_SKIP_DIRS = {".git", "__pycache__", ".github", "build", "dist", ".venv"}


def _walk(root: Path, source: Source) -> List[Path]:
    roots = [root / p for p in source.include] if source.include else [root]
    suffixes = (".py",) if source.language == "python" else (".c", ".h", ".cpp", ".hpp")
    out: List[Path] = []
    for base in roots:
        if not base.exists():
            logger.debug("include path %s missing in %s", base, source.name)
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            parts = set(path.parts)
            if parts & _SKIP_DIRS or parts & set(source.exclude):
                continue
            out.append(path)
    return out


# --------------------------------------------------------------------------
# content list
# --------------------------------------------------------------------------


def to_content_list(
    source: Source, checkout: Optional[Path], max_files: int = 120
) -> List[Dict[str, Any]]:
    """Render one source as a RAG-Anything content list.

    The overview and the capability table are always emitted, even when the
    checkout is missing, so that the corpus remains queryable ("what did we
    intend to learn from X?") even when a fetch failed.
    """
    page = 0
    items: List[Dict[str, Any]] = []

    overview = (
        f"Source project '{source.name}' ({source.kind}, {source.language}), "
        f"licensed {source.license}, at {source.url}. "
        f"Why it is in this corpus: {source.relevance}"
    )
    items.append({"type": "text", "text": overview, "page_idx": page})
    page += 1

    if source.provides:
        items.append(
            {
                "type": "table",
                "table_body": "\n".join(
                    ["| capability |", "| --- |"]
                    + [f"| {c} |" for c in source.provides]
                ),
                "table_caption": [f"Capabilities available from {source.name}"],
                "table_footnote": [
                    f"Licence {source.license}. Source {source.url}. "
                    "Verify the licence before depending on this in a product."
                ],
                "page_idx": page,
            }
        )
        page += 1

    if checkout is None:
        return items

    files = _walk(Path(checkout), source)
    if not files:
        logger.warning("no matching files found for %s under %s", source.name, checkout)
        return items

    root = Path(checkout)
    for path in files[:max_files]:
        if source.language == "python":
            structure = extract_python_structure(path)
        else:
            structure = _extract_c_header(path)
        if structure is None:
            continue

        rel = path.relative_to(root)
        lines = [f"File {rel} in project {source.name} ({structure['n_lines']} lines)."]
        if structure.get("module_doc"):
            lines.append(f"Purpose: {structure['module_doc']}")

        api = structure.get("api") or []
        if api:
            lines.append("Public API:")
            for entry in api[:40]:
                line = f"- {entry['signature']}"
                if entry.get("doc"):
                    line += f" -- {entry['doc']}"
                if entry.get("methods"):
                    line += f" [methods: {entry['methods']}]"
                lines.append(line)
        for signature in (structure.get("signatures") or [])[:30]:
            lines.append(f"- {signature}")

        items.append({"type": "text", "text": "\n".join(lines), "page_idx": page})
        page += 1

    if len(files) > max_files:
        items.append(
            {
                "type": "text",
                "text": (
                    f"Project {source.name} contains {len(files)} matching files; "
                    f"the first {max_files} were indexed. Raise max_files to widen "
                    "coverage."
                ),
                "page_idx": page,
            }
        )
    return items


# --------------------------------------------------------------------------
# ingestion
# --------------------------------------------------------------------------


async def ingest_corpus(
    rag: Any,
    sources: Optional[Sequence[Source]] = None,
    workdir: str = "./corpus_checkouts",
    max_files: int = 120,
    fetch: bool = True,
) -> Dict[str, int]:
    """Fetch and index the corpus into a RAG-Anything instance.

    Returns a mapping of source name to the number of content items indexed, so
    a caller can see at a glance which projects actually landed and which were
    unreachable.
    """
    sources = list(sources or CORPUS)
    root = Path(workdir)
    results: Dict[str, int] = {}

    for source in sources:
        checkout = fetch_source(source, root) if fetch else (root / source.name)
        if checkout is not None and not Path(checkout).exists():
            checkout = None

        content_list = to_content_list(source, checkout, max_files=max_files)
        if not content_list:
            results[source.name] = 0
            continue

        await rag.insert_content_list(
            content_list=content_list,
            file_path=f"{source.name}.corpus",
            doc_id=f"corpus:{source.name}",
        )
        results[source.name] = len(content_list)
        logger.info("indexed %s: %d items", source.name, len(content_list))
    return results
