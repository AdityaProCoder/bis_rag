from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from pypdf import PdfReader  # type: ignore[reportMissingImports]


SPACE_RE = re.compile(r"[ \t\u200b\ufeff]+")
SUMMARY_RE = re.compile(r"^SUMMARY\s+OF$", re.I)
HEADER_ONLY_RE = re.compile(r"^SP\s*21\s*:\s*2005$", re.I)
PAGE_NUM_RE = re.compile(r"^\d+\.\d+$")
ROMAN_RE = re.compile(r"^[ivxlcdm]+$", re.I)
STOP_RE = re.compile(
    r"^(?:\d+\.\s*|TABLE\b|NOTE\b|For detailed information\b|SUMMARY\s+OF\b)",
    re.I,
)
REVISION_RE = re.compile(r"^\(\s*(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)?\s*revis\w*\s*\)$", re.I)
INLINE_REVISION_RE = re.compile(
    r"\(\s*(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)?\s*revis\w*\s*\)",
    re.I,
)
ID_RE = re.compile(r"\b(\d{1,5})\b")
YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


@dataclass
class ChunkAnchor:
    page_index: int
    line_index: int
    kind: str


@dataclass
class ParsedRecord:
    id: str
    year: str
    title: str
    part: Optional[str]
    content: str
    confidence: str
    start_page: int
    header_mode: str


def normalize_line(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r"^\s*I\s*S\b", "IS", text, flags=re.I)
    return text.strip()


def clean_page_lines(raw_text: str) -> tuple[list[str], Optional[str]]:
    lines: list[str] = []
    page_label: Optional[str] = None
    for raw_line in raw_text.splitlines():
        line = normalize_line(raw_line)
        if not line:
            continue
        if PAGE_NUM_RE.fullmatch(line):
            page_label = line
            continue
        if HEADER_ONLY_RE.fullmatch(line):
            continue
        if ROMAN_RE.fullmatch(line) and len(line) <= 5:
            continue
        lines.append(line)
    return lines, page_label


def extract_pages(pdf_path: Path) -> tuple[list[list[str]], list[Optional[str]]]:
    reader = PdfReader(str(pdf_path))
    pages: list[list[str]] = []
    labels: list[Optional[str]] = []
    for page in reader.pages:
        lines, label = clean_page_lines(page.extract_text() or "")
        pages.append(lines)
        labels.append(label)
    return pages, labels


def is_summary_line(line: str) -> bool:
    return bool(SUMMARY_RE.fullmatch(normalize_line(line)))


def is_header_candidate(line: str) -> bool:
    return normalize_line(line).upper().startswith("IS")


def is_title_like(line: str) -> bool:
    stripped = normalize_line(line)
    if not stripped:
        return False
    if REVISION_RE.match(stripped):
        return True
    if stripped.startswith(("PART", "SEC", "SECTION")):
        return True
    if stripped.startswith("(") and "REVISION" in stripped.upper():
        return True
    if re.match(r"^[+*§†]+", stripped):
        return False
    if re.match(r"^\d+\.", stripped):
        return False
    letters = re.sub(r"[^A-Za-z]+", "", stripped)
    if not letters:
        return False
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio >= 0.55


def find_page_anchors(page_lines: list[str], window: int = 12) -> list[ChunkAnchor]:
    anchors: list[ChunkAnchor] = []
    summary_positions = [idx for idx, line in enumerate(page_lines) if is_summary_line(line)]
    for summary_idx in summary_positions:
        back_candidates = [
            idx
            for idx in range(max(0, summary_idx - window), summary_idx)
            if is_header_candidate(page_lines[idx])
            and parse_header_line(page_lines[idx]) is not None
            and any(
                is_title_like(page_lines[mid])
                for mid in range(idx + 1, summary_idx)
                if not REVISION_RE.match(page_lines[mid])
                and not normalize_line(page_lines[mid]).upper().startswith("FOR DETAILED INFORMATION")
            )
        ]
        if back_candidates:
            anchors.append(ChunkAnchor(page_index=-1, line_index=back_candidates[-1], kind="header_before_summary"))
            continue

        forward_candidates = [
            idx
            for idx in range(summary_idx + 1, min(len(page_lines), summary_idx + window + 1))
            if is_header_candidate(page_lines[idx])
        ]
        if forward_candidates:
            anchors.append(ChunkAnchor(page_index=-1, line_index=summary_idx, kind="summary_before_header"))
    return anchors


def flatten_with_page_map(pages: list[list[str]]) -> tuple[list[str], list[int]]:
    all_lines: list[str] = []
    line_pages: list[int] = []
    for page_index, page_lines in enumerate(pages):
        for line in page_lines:
            all_lines.append(line)
            line_pages.append(page_index)
    return all_lines, line_pages


def build_chunk_anchors(pages: list[list[str]]) -> list[ChunkAnchor]:
    anchors: list[ChunkAnchor] = []
    for page_index, page_lines in enumerate(pages):
        for anchor in find_page_anchors(page_lines):
            anchors.append(
                ChunkAnchor(
                    page_index=page_index,
                    line_index=anchor.line_index,
                    kind=anchor.kind,
                )
            )
    # Sort anchors in reading order.
    anchors.sort(key=lambda a: (a.page_index, a.line_index))
    return anchors


def normalize_part(part_raw: str) -> str:
    part = part_raw.strip().strip("()[]{}:;,-")
    part = SPACE_RE.sub(" ", part)
    part = re.sub(r"\b(PART|SEC|SECTION)([IVX]+)\b", r"\1 \2", part, flags=re.I)
    part = re.sub(r"\b(PART|SEC|SECTION)(\d+)\b", r"\1 \2", part, flags=re.I)
    part = re.sub(r"\bPART\s*([IVX]+)\b", r"Part \1", part, flags=re.I)
    part = re.sub(r"\bSEC\s*([IVX]+)\b", r"Sec \1", part, flags=re.I)
    part = re.sub(r"\bSECTION\s*([IVX]+)\b", r"Section \1", part, flags=re.I)
    part = re.sub(r"\bPART\s*(\d)", r"Part \1", part, flags=re.I)
    part = re.sub(r"\bSEC\s*(\d)", r"Sec \1", part, flags=re.I)
    part = re.sub(r"\bSECTION\s*(\d)", r"Section \1", part, flags=re.I)
    part = re.sub(r"\bAND\b", "and", part, flags=re.I)
    part = re.sub(r"\s*/\s*", "/", part)
    part = re.sub(r"\s+", " ", part).strip()
    return part


def parse_header_line(line: str) -> Optional[dict]:
    s = normalize_line(line)
    if not s.upper().startswith("IS"):
        return None

    body = re.sub(r"^IS\b\s*:?", "", s, flags=re.I).strip()
    id_match = ID_RE.search(body)
    if not id_match:
        return None

    std_id = id_match.group(1)
    after_id = body[id_match.end():]
    year_match = YEAR_RE.search(after_id)
    if not year_match:
        return None

    year = year_match.group(1)
    pre_year = after_id[:year_match.start()].strip()
    title_seed = after_id[year_match.end():].strip()

    part: Optional[str] = None
    paren_part = re.search(r"\(\s*((?:PART|SEC|SECTION)\s*[^)]*?)\s*\)", pre_year, re.I)
    bare_part = re.search(r"\b((?:PART|SEC|SECTION)\s*[A-Z0-9/ .-]+?)(?:\b|$)", pre_year, re.I)
    part_match = paren_part or bare_part
    if part_match:
        part = normalize_part(part_match.group(1))

    header_mode = "simple"
    if re.search(r"^IS\s*:\s*\d", s, re.I):
        header_mode = "colon_after_is"
    if re.search(r"\b\d{1,5}\s*-\s*(?:19|20)\d{2}\b", s):
        header_mode = "hyphen_year"
    if part and not re.search(r"\(\s*(?:PART|SEC|SECTION)", pre_year, re.I):
        header_mode = "bare_part"

    return {
        "id": f"IS {std_id}",
        "year": year,
        "part": part,
        "title_seed": title_seed,
        "header_mode": header_mode,
    }


def title_looks_invalid(title: str) -> bool:
    upper = title.upper().strip()
    if not upper:
        return True
    if upper == "UNKNOWN":
        return True
    if upper.startswith("NOTE "):
        return True
    if upper.startswith("TABLE "):
        return True
    return False


def extract_title(chunk_lines: list[str], header_idx: int, seed_title: str) -> str:
    parts: list[str] = []
    seed_title = seed_title.strip()
    if seed_title and not re.fullmatch(r"[+*§†.]+", seed_title):
        parts.append(seed_title)

    for line in chunk_lines[header_idx + 1 :]:
        if STOP_RE.match(line):
            break
        if REVISION_RE.match(line):
            continue
        if not is_title_like(line):
            break
        parts.append(line)

    title = " ".join(parts)
    title = INLINE_REVISION_RE.sub("", title)
    title = re.sub(r"^[\s\.\+\*§†\-–—:;]+", "", title)
    title = re.sub(r"\s+", " ", title).strip(" -:;")
    return title


def build_records(pages: list[list[str]]) -> tuple[list[ParsedRecord], dict]:
    anchors = build_chunk_anchors(pages)
    all_lines, line_pages = flatten_with_page_map(pages)

    # Convert anchors to global line indices.
    page_offsets: list[int] = []
    cursor = 0
    for page_lines in pages:
        page_offsets.append(cursor)
        cursor += len(page_lines)

    global_starts: list[int] = []
    for anchor in anchors:
        global_starts.append(page_offsets[anchor.page_index] + anchor.line_index)

    # Remove duplicate starts while preserving order.
    deduped_starts: list[int] = []
    for idx in global_starts:
        if not deduped_starts or deduped_starts[-1] != idx:
            deduped_starts.append(idx)

    records: list[ParsedRecord] = []
    parse_failures: list[list[str]] = []

    for pos, start in enumerate(deduped_starts):
        end = deduped_starts[pos + 1] if pos + 1 < len(deduped_starts) else len(all_lines)
        chunk_lines = all_lines[start:end]
        chunk_page = line_pages[start] + 1

        header_idx = None
        header = None
        summary_idx = next((idx for idx, line in enumerate(chunk_lines) if is_summary_line(line)), None)
        if summary_idx is not None:
            back_candidates = [
                idx
                for idx in range(max(0, summary_idx - 12), summary_idx)
                if is_header_candidate(chunk_lines[idx])
                and parse_header_line(chunk_lines[idx]) is not None
                and any(
                    is_title_like(chunk_lines[mid])
                    for mid in range(idx + 1, summary_idx)
                    if not REVISION_RE.match(chunk_lines[mid])
                    and not normalize_line(chunk_lines[mid]).upper().startswith("FOR DETAILED INFORMATION")
                )
            ]
            if back_candidates:
                header_idx = back_candidates[-1]
                header = parse_header_line(chunk_lines[header_idx])
            else:
                for idx in range(summary_idx + 1, min(len(chunk_lines), summary_idx + 13)):
                    if is_header_candidate(chunk_lines[idx]):
                        parsed = parse_header_line(chunk_lines[idx])
                        if parsed:
                            header_idx = idx
                            header = parsed
                            break
        else:
            search_limit = min(len(chunk_lines), 14)
            for idx in range(search_limit):
                if is_header_candidate(chunk_lines[idx]):
                    parsed = parse_header_line(chunk_lines[idx])
                    if parsed:
                        header_idx = idx
                        header = parsed
                        break

        if header is None or header_idx is None:
            parse_failures.append(chunk_lines[:12])
            continue

        title = extract_title(chunk_lines, header_idx, header["title_seed"])
        confidence = "high"
        if header["header_mode"] != "simple":
            confidence = "medium"
        if len(chunk_lines) < 6 or len(" ".join(chunk_lines)) < 180 or len(title) < 8:
            confidence = "low" if confidence == "high" else confidence
        if title_looks_invalid(title):
            confidence = "low"

        record = ParsedRecord(
            id=header["id"],
            year=header["year"],
            title=title,
            part=header["part"],
            content="\n".join(chunk_lines).strip(),
            confidence=confidence,
            start_page=chunk_page,
            header_mode=header["header_mode"],
        )
        records.append(record)

    stats = {
        "parse_failures": parse_failures,
        "total_anchors": len(deduped_starts),
        "parsed_records": len(records),
    }
    return records, stats


def dedupe_records(records: list[ParsedRecord]) -> tuple[list[ParsedRecord], int]:
    best: dict[tuple[str, str, Optional[str]], ParsedRecord] = {}
    duplicates = 0
    for record in records:
        key = (record.id, record.year, record.part)
        existing = best.get(key)
        if existing is None:
            best[key] = record
            continue
        duplicates += 1
        if len(record.content) > len(existing.content):
            best[key] = record
    deduped = sorted(best.values(), key=lambda r: (r.start_page, r.id, r.year, r.part or ""))
    return deduped, duplicates


DETAIL_REF_RE = re.compile(
    r"^For detailed information,\s*refer to\s*IS\s*:?\s*(?P<id>\d{1,5})"
    r"(?:\s*\(\s*(?P<part>PART|SEC|SECTION)\s*(?P<part_num>[A-Z0-9/ .-]+?)\s*\))?"
    r"(?:\s*:\s*(?P<year>(?:19|20)\d{2}))?",
    re.I,
)


def clean_reference_title(line: str, ref_match: re.Match[str]) -> str:
    title = line.strip()
    title = re.sub(r"^For detailed information,\s*refer to\s*", "", title, flags=re.I)
    title = re.sub(
        rf"^IS\s*:?\s*{re.escape(ref_match.group('id'))}"
        rf"(?:\s*\(\s*(?:PART|SEC|SECTION)\s*[A-Z0-9/ .-]+?\s*\))?"
        rf"(?:\s*:\s*{re.escape(ref_match.group('year') or '')})?\s*",
        "",
        title,
        flags=re.I,
    )
    title = re.sub(r"^Specification for\s+", "", title, flags=re.I)
    title = INLINE_REVISION_RE.sub("", title)
    title = re.sub(r"^[\s\.\+\*§†\-–—:;]+", "", title)
    title = re.sub(r"\s+", " ", title).strip(" -:;.")
    return title.upper()


def split_embedded_records(records: list[ParsedRecord]) -> list[ParsedRecord]:
    """Split records that accidentally swallow the next standard inside their content."""

    def parse_embedded_header(lines: list[str]) -> Optional[dict]:
        for idx, line in enumerate(lines):
            if is_summary_line(line):
                for j in range(idx + 1, min(len(lines), idx + 9)):
                    parsed = parse_header_line(lines[j])
                    if parsed:
                        return {**parsed, "header_idx": j}
        return None

    def maybe_split(record: ParsedRecord) -> list[ParsedRecord]:
        lines = record.content.splitlines()
        for idx, line in enumerate(lines):
            m = DETAIL_REF_RE.match(line.strip())
            if not m:
                continue
            if f"IS {m.group('id')}" == record.id and (m.group("year") or record.year) == record.year:
                continue
            # Only split when the following lines clearly look like the start of a new standard.
            tail = lines[idx + 1 : idx + 6]
            if not any(re.match(r"^(?:1\.\s*Scope\b|1\s*Scope\b|1\.\b|\d+\.\s*Scope\b)", t, re.I) for t in tail):
                continue

            right_lines = lines[idx + 1 :]
            left_lines = lines[:idx]
            if len(right_lines) < 20 or len(left_lines) < 20:
                continue

            ref_id = f"IS {m.group('id')}"
            ref_year = m.group("year") or record.year
            ref_part = None
            if m.group("part"):
                ref_part = normalize_part(f"{m.group('part')} {m.group('part_num') or ''}".strip())

            left_record = ParsedRecord(
                id=record.id,
                year=record.year,
                title=record.title,
                part=record.part,
                content="\n".join(left_lines).strip(),
                confidence=record.confidence,
                start_page=record.start_page,
                header_mode=record.header_mode,
            )

            right_title = clean_reference_title(line, m)
            right_conf = "medium"
            if len(right_lines) < 6 or len(" ".join(right_lines)) < 180 or len(right_title) < 8:
                right_conf = "low" if right_conf == "high" else right_conf
            if title_looks_invalid(right_title):
                right_conf = "low"
            right_record = ParsedRecord(
                id=ref_id,
                year=ref_year,
                title=right_title,
                part=ref_part,
                content="\n".join(right_lines).strip(),
                confidence=right_conf,
                start_page=record.start_page,
                header_mode="reference_split",
            )
            return [left_record, right_record]
        return [record]

    output: list[ParsedRecord] = []
    for record in records:
        output.extend(maybe_split(record))
    return output


def serialize(records: list[ParsedRecord]) -> list[dict]:
    payload = []
    for record in records:
        payload.append(
            {
                "id": record.id,
                "year": record.year,
                "title": record.title,
                "part": record.part,
                "content": record.content,
                "confidence": record.confidence,
            }
        )
    return payload


def print_report(records: list[ParsedRecord], duplicates: int, parse_failures: list[list[str]]) -> None:
    total = len(records) + duplicates
    if total == 0:
        print("No records parsed.")
        return

    title_valid = sum(1 for r in records if not title_looks_invalid(r.title))
    id_valid = sum(1 for r in records if bool(re.fullmatch(r"IS \d{1,5}", r.id)))
    low_conf = sum(1 for r in records if r.confidence == "low")

    print(f"Total parsed standards: {len(records)}")
    print(f"Total duplicates removed: {duplicates}")
    print(f"Valid ID rate: {id_valid / len(records) * 100:.2f}%")
    print(f"Valid title rate: {title_valid / len(records) * 100:.2f}%")
    print(f"Low confidence rate: {low_conf / len(records) * 100:.2f}%")
    print(f"Parse failures: {len(parse_failures)}")

    if parse_failures:
        print("\nSample parse failures:")
        for chunk in parse_failures[:3]:
            print(" - " + " | ".join(chunk[:6]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Parse BIS SP 21:2005 handbook into structured JSON.")
    parser.add_argument("--input", default="data/dataset_ocr.pdf", help="Input PDF path.")
    parser.add_argument("--output", default="data/sp21_standards.json", help="Output JSON path.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for optional sample inspection.")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to print.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    pages, page_labels = extract_pages(input_path)
    records, stats = build_records(pages)
    records = split_embedded_records(records)
    records, duplicates = dedupe_records(records)

    # Print a compact report before writing the dataset.
    print_report(records, duplicates, stats["parse_failures"])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(serialize(records), f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(records)} records to {output_path}")

    random.seed(args.seed)
    if records and args.samples > 0:
        print("\nRandom samples:")
        for sample in random.sample(records, min(args.samples, len(records))):
            print(f"- {sample.id} {sample.year} {sample.part or ''}".rstrip())
            print(f"  {sample.title}")
            preview = sample.content[:260].replace("\n", " | ")
            print(f"  {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
