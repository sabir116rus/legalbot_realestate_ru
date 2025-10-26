"""Utility for analyzing LegalBot conversation logs.

The script parses a CSV log file, computes descriptive statistics, infers topics
through the knowledge base, and prints readable summaries for operators.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from rag import KnowledgeBase


def parse_args() -> argparse.Namespace:
    """Configure CLI arguments."""
    default_log = Path(__file__).resolve().parent / "data/log.csv"
    parser = argparse.ArgumentParser(description="Analyze LegalBot logs")
    parser.add_argument(
        "--log",
        type=Path,
        default=default_log,
        help="Path to CSV log file (default: %(default)s)",
    )
    parser.add_argument(
        "--knowledge",
        type=str,
        default="data/knowledge.json",
        help="Path to knowledge base JSON used for topic inference",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of documents to request from the knowledge base",
    )
    parser.add_argument(
        "--low-score-threshold",
        type=float,
        default=70.0,
        help="Threshold for flagging weak answers by top_score",
    )
    return parser.parse_args()


def load_logs(path: Path) -> pd.DataFrame:
    """Load logs from CSV and normalise columns."""
    if not path.exists() or path.is_dir():
        print(f"[!] Log file not found: {path}")
        sys.exit(1)

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - pandas error message is enough
        print(f"[!] Unable to read log CSV: {exc}")
        sys.exit(1)

    if df.empty:
        print(f"[!] Log file is empty: {path}")
        sys.exit(1)

    df["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    df["top_score"] = pd.to_numeric(df.get("top_score"), errors="coerce")
    df["tokens"] = pd.to_numeric(df.get("tokens"), errors="coerce")
    return df


def infer_topics(
    df: pd.DataFrame,
    knowledge_path: Path,
    top_k: int,
) -> pd.DataFrame:
    """Infer topics for questions using the KnowledgeBase."""
    if top_k <= 0:
        top_k = 1

    try:
        kb = KnowledgeBase(str(knowledge_path))
    except Exception as exc:
        print(f"[!] Unable to load knowledge base ({knowledge_path}): {exc}")
        print("[i] Proceeding without topic inference.")
        df["inferred_topic"] = pd.NA
        df["inferred_score"] = pd.NA
        return df

    cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

    topics = []
    scores = []
    for question in df.get("question", []):
        q = question if isinstance(question, str) else ""
        key = q.strip()
        if key not in cache:
            results = kb.query(key, top_k=top_k) if key else []
            if results:
                best = results[0]
                cache[key] = (best.get("topic"), best.get("score"))
            else:
                cache[key] = (None, None)
        topic, score = cache[key]
        topics.append(topic)
        scores.append(score)

    df = df.copy()
    df["inferred_topic"] = topics
    df["inferred_score"] = scores
    return df


def format_series(series: pd.Series, limit: Optional[int] = None) -> str:
    if series.empty:
        return "(нет данных)"
    if limit is not None:
        series = series.head(limit)
    return series.to_string()


def format_dataframe(frame: pd.DataFrame, limit: Optional[int] = None) -> str:
    if frame.empty:
        return "(нет данных)"
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_string(index=False)


def build_summary(df: pd.DataFrame, threshold: float) -> str:
    records = len(df)
    users = df["user_id"].nunique(dropna=True) if "user_id" in df else 0
    avg_score = df["top_score"].mean()
    avg_tokens = df["tokens"].mean()
    errors = (df["status"].fillna("") != "ok").sum() if "status" in df else 0

    summary_lines = [
        f"Всего записей: {records}",
        f"Активных пользователей: {users}",
        f"Средний top_score: {avg_score:.2f}" if pd.notna(avg_score) else "Средний top_score: н/д",
        f"Средняя длина ответа (tokens): {avg_tokens:.2f}" if pd.notna(avg_tokens) else "Средняя длина ответа (tokens): н/д",
        f"Число ошибок (status != 'ok'): {errors}",
        f"Порог слабых ответов: {threshold}",
    ]
    return "\n".join(summary_lines)


def _resolve_path(candidate: Path, base: Path) -> Path:
    candidate = Path(candidate).expanduser()
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    log_path = _resolve_path(args.log, script_dir)
    knowledge_path = _resolve_path(Path(args.knowledge), script_dir)

    df = load_logs(log_path)
    df = infer_topics(df, knowledge_path, args.top_k)

    summary = build_summary(df, args.low_score_threshold)

    by_model = df.groupby("model").size().sort_values(ascending=False) if "model" in df else pd.Series(dtype=int)
    by_status = df.groupby("status").size().sort_values(ascending=False) if "status" in df else pd.Series(dtype=int)

    if df["timestamp"].notna().any():
        df_time = df.dropna(subset=["timestamp"]).copy()
        df_time.set_index("timestamp", inplace=True)
        by_day = df_time.resample("D").size()
        by_hour = df_time.groupby(df_time.index.hour).size().sort_index()
    else:
        by_day = pd.Series(dtype=int)
        by_hour = pd.Series(dtype=int)

    topic_counts = (
        df["inferred_topic"].fillna("(не определено)").value_counts().sort_values(ascending=False)
        if "inferred_topic" in df
        else pd.Series(dtype=int)
    )
    topic_avg_score = (
        df.groupby("inferred_topic")["top_score"].mean().sort_values(ascending=False)
        if "inferred_topic" in df
        else pd.Series(dtype=float)
    )

    top_users = (
        df.groupby(["user_id", "username"], dropna=False)
        .size()
        .reset_index(name="requests")
        .sort_values("requests", ascending=False)
    )

    low_score_questions = df[df["top_score"] <= args.low_score_threshold].sort_values("top_score")
    low_score_columns = [c for c in ["timestamp", "user_id", "username", "question", "top_score", "model"] if c in df.columns]

    print("=== Общая статистика ===")
    print(summary)

    print("\n=== Распределение по моделям ===")
    print(format_series(by_model))

    print("\n=== Распределение по статусам ===")
    print(format_series(by_status))

    print("\n=== Динамика по дням ===")
    print(format_series(by_day))

    print("\n=== Динамика по часам ===")
    print(format_series(by_hour))

    print("\n=== Топ-5 тем ===")
    print(format_series(topic_counts, limit=5))

    print("\n=== Средний top_score по темам ===")
    print(format_series(topic_avg_score))

    print("\n=== Топ-10 пользователей по числу запросов ===")
    print(format_dataframe(top_users, limit=10))

    print("\n=== Запросы с низким top_score ===")
    if low_score_questions.empty:
        print("(нет запросов ниже порога)")
    else:
        print(format_dataframe(low_score_questions[low_score_columns], limit=10))


if __name__ == "__main__":
    main()
