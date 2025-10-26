"""Convert legacy CSV knowledge base into JSON format."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse

from models.knowledge import KnowledgeRecord, Source

URL_RE = re.compile(r"https?://[^\s,]+", re.IGNORECASE)


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def split_list(value: str) -> List[str]:
    if not value:
        return []
    parts = re.split(r"[;\n]+", value)
    cleaned = []
    for part in parts:
        normalized = re.sub(r"\s+", " ", part).strip()
        if normalized:
            cleaned.append(normalized)
    return cleaned


def parse_sources(raw_value: str) -> List[Source]:
    if not raw_value:
        return []

    entries: List[Source] = []
    consumed_urls: set[str] = set()

    for match in URL_RE.finditer(raw_value):
        url = match.group(0).rstrip(").")
        if url in consumed_urls:
            continue
        consumed_urls.add(url)
        netloc = urlparse(url).netloc or url
        entries.append(Source(title=netloc, url=url))

    # Remove URLs from the raw text to collect remaining descriptions.
    without_urls = URL_RE.sub("", raw_value)
    for fragment in split_list(without_urls):
        if fragment.lower() in {source.title.lower() for source in entries}:
            continue
        entries.append(Source(title=fragment))

    return entries


def convert_records(rows: Iterable[dict]) -> List[KnowledgeRecord]:
    records: List[KnowledgeRecord] = []
    for index, row in enumerate(rows, start=1):
        raw_id = (row.get("id") or "").strip()
        record_id = raw_id or f"auto_{index}"
        if raw_id:
            try:
                numeric = float(raw_id)
            except ValueError:
                pass
            else:
                if numeric.is_integer():
                    record_id = str(int(numeric))
        law_refs_text = (row.get("law_refs") or "").strip()
        law_refs_norm = split_list(law_refs_text)
        sources_raw = (row.get("url") or "").strip()
        sources = parse_sources(sources_raw)

        topic_value = (row.get("topic") or "").strip() or "Нормативные акты"

        record = KnowledgeRecord(
            id=record_id,
            topic=topic_value,
            question=row.get("question"),
            answer=row.get("answer"),
            law_refs_text=law_refs_text,
            law_refs_norm=law_refs_norm,
            sources=sources,
        )
        records.append(record)
    return records


def write_json(records: List[KnowledgeRecord], path: Path) -> None:
    payload = [record.model_dump_for_storage() for record in records]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def migrate(csv_path: Path, json_path: Path) -> None:
    rows = read_csv(csv_path)
    records = convert_records(rows)
    write_json(records, json_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="Путь к исходному knowledge.csv")
    parser.add_argument("json_path", type=Path, help="Целевой файл knowledge.json")
    args = parser.parse_args()

    migrate(args.csv_path, args.json_path)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
