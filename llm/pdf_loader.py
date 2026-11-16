"""PDF loading utilities.

The default project flow now prefers OpenDataLoader for structured PDF-to-Markdown
conversion, while keeping pypdf as a fallback for environments without Java.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from pypdf import PdfReader


def load_pdfs(raw_dir: str, config: dict[str, Any] | None = None) -> str:
    """Extract and concatenate text from all PDFs in raw_dir.

    Falls back to reading `*.md`/`*.markdown` files verbatim when no PDFs are
    present — e.g. a pre-parsed or already-masked policy document that skips
    the PDF-to-markdown conversion step entirely.
    """
    pdf_paths = sorted(Path(raw_dir).glob("*.pdf"))
    if not pdf_paths:
        md_paths = sorted(Path(raw_dir).glob("*.md")) + sorted(Path(raw_dir).glob("*.markdown"))
        if not md_paths:
            raise FileNotFoundError(f"No PDF or Markdown files found in {raw_dir}")
        texts = [
            _format_source_text(md_path, md_path.read_text(encoding="utf-8"), {"parser": "raw_markdown", "format": "markdown"})
            for md_path in md_paths
        ]
        return "\n\n---\n\n".join(texts)

    pdf_config = _pdf_config(config)
    texts = []
    for pdf_path in pdf_paths:
        pages_text, metadata = _load_single_pdf(pdf_path, pdf_config, config)
        texts.append(_format_source_text(pdf_path, pages_text, metadata))

    return "\n\n---\n\n".join(texts)


def _load_single_pdf(
    pdf_path: Path,
    pdf_config: dict[str, Any],
    config: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    parser = str(pdf_config.get("parser", "pypdf")).lower()
    if parser in {"opendataloader", "open_data_loader", "odl"}:
        try:
            return _extract_with_opendataloader(pdf_path, pdf_config, config)
        except Exception as exc:
            if not pdf_config.get("fallback_to_pypdf", True):
                raise
            text, metadata = _extract_with_pypdf(pdf_path)
            metadata["parser"] = "pypdf_fallback"
            metadata["fallback_reason"] = str(exc)
            return text, metadata

    if parser == "pypdf":
        return _extract_with_pypdf(pdf_path)

    raise ValueError(f"Unsupported PDF parser: {parser}")


def _extract_with_pypdf(pdf_path: Path) -> tuple[str, dict[str, Any]]:
    reader = PdfReader(str(pdf_path))
    pages_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return pages_text, {
        "parser": "pypdf",
        "page_count": len(reader.pages),
        "format": "text",
    }


def _extract_with_opendataloader(
    pdf_path: Path,
    pdf_config: dict[str, Any],
    config: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    _ensure_java_on_path(pdf_config)

    try:
        import opendataloader_pdf
    except ImportError as exc:
        raise RuntimeError(
            "OpenDataLoader parser requested but opendataloader-pdf is not installed."
        ) from exc

    cache_root = _cache_root(pdf_config, config)
    output_dir = cache_root / "opendataloader" / _cache_key(pdf_path, pdf_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    markdown_path = _find_output_file(output_dir, pdf_path.stem, (".md", ".markdown"))
    json_path = _find_output_file(output_dir, pdf_path.stem, (".json",))

    if not markdown_path:
        convert_kwargs = _opendataloader_kwargs(pdf_config)
        opendataloader_pdf.convert(
            input_path=[str(pdf_path)],
            output_dir=str(output_dir),
            **convert_kwargs,
        )
        markdown_path = _find_output_file(output_dir, pdf_path.stem, (".md", ".markdown"))
        json_path = _find_output_file(output_dir, pdf_path.stem, (".json",))

    if not markdown_path:
        text_path = _find_output_file(output_dir, pdf_path.stem, (".txt",))
        if text_path:
            text = text_path.read_text(encoding="utf-8", errors="replace")
            return text, {
                "parser": "opendataloader",
                "format": "text",
                "cache_dir": str(output_dir),
                "text_path": str(text_path),
                "json_path": str(json_path) if json_path else None,
            }
        raise RuntimeError(f"OpenDataLoader did not produce Markdown output in {output_dir}")

    text = markdown_path.read_text(encoding="utf-8", errors="replace")
    return text, {
        "parser": "opendataloader",
        "format": "markdown",
        "cache_dir": str(output_dir),
        "markdown_path": str(markdown_path),
        "json_path": str(json_path) if json_path else None,
    }


def _opendataloader_kwargs(pdf_config: dict[str, Any]) -> dict[str, Any]:
    formats = pdf_config.get("output_formats", ["markdown", "json"])
    if isinstance(formats, (list, tuple)):
        format_value = ",".join(str(item) for item in formats)
    else:
        format_value = str(formats)

    kwargs: dict[str, Any] = {
        "format": format_value,
        "quiet": bool(pdf_config.get("quiet", True)),
    }
    optional_keys = (
        "use_struct_tree",
        "table_method",
        "reading_order",
        "keep_line_breaks",
        "include_header_footer",
        "markdown_page_separator",
        "content_safety_off",
        "threads",
    )
    for key in optional_keys:
        value = pdf_config.get(key)
        if value is not None:
            kwargs[key] = value
    return kwargs


def _pdf_config(config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = {
        "parser": "pypdf",
        "fallback_to_pypdf": True,
        "output_formats": ["markdown", "json"],
        "quiet": True,
    }
    if config and isinstance(config.get("pdf"), dict):
        defaults.update(config["pdf"])
    return defaults


def _cache_root(pdf_config: dict[str, Any], config: dict[str, Any] | None) -> Path:
    if pdf_config.get("cache_dir"):
        return Path(pdf_config["cache_dir"])
    if config:
        processed_dir = config.get("paths", {}).get("processed_dir")
        if processed_dir:
            return Path(processed_dir) / "pdf_cache"
    return Path("dataset/processed/pdf_cache")


def _cache_key(pdf_path: Path, pdf_config: dict[str, Any]) -> str:
    stat = pdf_path.stat()
    relevant_config = {
        key: pdf_config.get(key)
        for key in (
            "output_formats",
            "use_struct_tree",
            "table_method",
            "reading_order",
            "keep_line_breaks",
            "include_header_footer",
            "markdown_page_separator",
        )
    }
    basis = {
        "path": str(pdf_path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "config": relevant_config,
    }
    digest = hashlib.sha1(json.dumps(basis, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{pdf_path.stem}_{digest}"


def _find_output_file(output_dir: Path, stem: str, suffixes: tuple[str, ...]) -> Path | None:
    exact_candidates = [output_dir / f"{stem}{suffix}" for suffix in suffixes]
    for candidate in exact_candidates:
        if candidate.exists():
            return candidate

    for suffix in suffixes:
        matches = sorted(output_dir.glob(f"*{suffix}"))
        if matches:
            return matches[0]
    return None


def _ensure_java_on_path(pdf_config: dict[str, Any]) -> None:
    java_home = pdf_config.get("java_home")
    if java_home:
        java_bin = Path(java_home) / "bin"
        _prepend_java(java_home, java_bin)

    if shutil.which("java"):
        return

    local_java = _find_local_java()
    if local_java:
        _prepend_java(local_java.parent.parent, local_java.parent)
        return

    if not shutil.which("java"):
        raise RuntimeError(
            "OpenDataLoader requires Java 11+ on PATH. Install a JDK or set pdf.java_home."
        )


def _prepend_java(java_home: Path | str, java_bin: Path | str) -> None:
    os.environ["JAVA_HOME"] = str(java_home)
    os.environ["PATH"] = str(java_bin) + os.pathsep + os.environ.get("PATH", "")


def _find_local_java() -> Path | None:
    repo_root = Path(__file__).resolve().parents[1]
    search_roots = [repo_root / ".tools", Path.cwd() / ".tools"]
    executable = "java.exe" if os.name == "nt" else "java"
    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.glob(f"**/bin/{executable}"))
        if matches:
            return matches[0]
    return None


def _format_source_text(pdf_path: Path, text: str, metadata: dict[str, Any]) -> str:
    header = [
        f"[Source: {pdf_path.name}]",
        f"[Parser: {metadata.get('parser', 'unknown')}]",
        f"[Format: {metadata.get('format', 'text')}]",
    ]
    if metadata.get("markdown_path"):
        header.append(f"[Parsed Markdown: {metadata['markdown_path']}]")
    if metadata.get("json_path"):
        header.append(f"[Parsed JSON: {metadata['json_path']}]")
    if metadata.get("fallback_reason"):
        header.append(f"[Parser fallback reason: {metadata['fallback_reason']}]")
    return "\n".join(header) + "\n" + text
