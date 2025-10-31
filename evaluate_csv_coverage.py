"""Evaluate knowledge coverage using CSV logs."""
from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import Config
from rag import KnowledgeBase
from services.google_drive_client import GoogleDriveClient

# A simple list of common Russian stop words to be excluded from keyword stats.
STOP_WORDS = {
    "и",
    "в",
    "во",
    "не",
    "что",
    "он",
    "на",
    "я",
    "с",
    "со",
    "как",
    "а",
    "то",
    "все",
    "она",
    "так",
    "его",
    "но",
    "да",
    "ты",
    "к",
    "у",
    "же",
    "вы",
    "за",
    "бы",
    "по",
    "ее",
    "из",
    "мы",
    "они",
    "тут",
    "при",
    "это",
    "них",
    "ли",
    "или",
    "же",
    "без",
    "если",
    "когда",
    "для",
    "про",
    "там",
    "вот",
    "от",
    "быть",
    "уже",
    "кто",
    "чтобы",
    "этот",
    "эти",
    "этом",
    "этим",
    "либо",
    "где",
    "куда",
    "зачем",
    "почему",
    "какой",
    "какая",
    "какие",
    "какого",
    "какому",
    "каком",
    "какую",
    "каких",
    "каким",
    "какими",
    "какую",
    "там",
    "сюда",
    "тогда",
    "такой",
    "такая",
    "такие",
    "ка",
    "лишь",
    "еще",
    "из-за",
}

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)


class CoverageEvaluator:
    """Compute coverage metrics for CSV logs."""

    def __init__(
        self,
        log_path: Path,
        knowledge_path: Path,
        threshold: float,
        use_stored_score: bool,
        top_n: int,
    ) -> None:
        self.log_path = log_path.expanduser()
        self.knowledge_path = knowledge_path.expanduser()
        self.threshold = threshold
        self.use_stored_score = use_stored_score
        self.top_n = max(1, top_n)

        self.df: pd.DataFrame = pd.DataFrame()
        self.uncovered_df: pd.DataFrame = pd.DataFrame()
        self.keyword_counter: Counter[str] = Counter()

    def load(self) -> None:
        if not self.log_path.exists() or self.log_path.is_dir():
            raise FileNotFoundError(f"Log CSV not found: {self.log_path}")
        self.df = pd.read_csv(self.log_path)
        if self.df.empty:
            raise ValueError("Log CSV is empty")

        questions = self.df.get("question")
        if questions is None:
            self.df["question"] = ""
        else:
            self.df["question"] = questions.fillna("").astype(str)

        stored_scores = pd.to_numeric(self.df.get("top_score"), errors="coerce")
        self.df["stored_score"] = stored_scores

    def _evaluate_scores(self) -> None:
        if self.use_stored_score:
            self.df["evaluated_score"] = self.df["stored_score"]
            topic_col = next(
                (col for col in ["topic", "inferred_topic", "top_topic"] if col in self.df.columns),
                None,
            )
            if topic_col:
                self.df["evaluated_topic"] = self.df[topic_col]
            else:
                self.df["evaluated_topic"] = pd.Series(pd.NA, index=self.df.index)

            kb_question_col = next(
                (col for col in ["kb_question", "matched_question", "retrieved_question"] if col in self.df.columns),
                None,
            )
            if kb_question_col:
                self.df["evaluated_kb_question"] = self.df[kb_question_col]
            else:
                self.df["evaluated_kb_question"] = pd.Series(pd.NA, index=self.df.index)
            return

        kb = KnowledgeBase(str(self.knowledge_path))
        cache: Dict[str, Tuple[Optional[float], Optional[str], Optional[str]]] = {}

        scores: List[Optional[float]] = []
        topics: List[Optional[str]] = []
        kb_questions: List[Optional[str]] = []

        for question in self.df["question"]:
            query = question.strip()
            if query not in cache:
                if not query:
                    cache[query] = (math.nan, None, None)
                else:
                    results = kb.query(query, top_k=1)
                    if results:
                        best = results[0]
                        cache[query] = (
                            float(best.get("score")) if best.get("score") is not None else math.nan,
                            best.get("topic"),
                            best.get("question"),
                        )
                    else:
                        cache[query] = (math.nan, None, None)
            score, topic, kb_question = cache[query]
            scores.append(score)
            topics.append(topic)
            kb_questions.append(kb_question)

        self.df["evaluated_score"] = scores
        self.df["evaluated_topic"] = topics
        self.df["evaluated_kb_question"] = kb_questions

    def _identify_uncovered(self) -> None:
        eval_scores = pd.to_numeric(self.df["evaluated_score"], errors="coerce")
        mask = eval_scores < self.threshold
        mask |= eval_scores.isna()
        self.uncovered_df = self.df.loc[mask].copy()

    def _collect_keywords(self) -> None:
        counter: Counter[str] = Counter()
        for question in self.uncovered_df["question"]:
            tokens = tokenize(question)
            counter.update(tokens)
        self.keyword_counter = counter

    def evaluate(self) -> None:
        self._evaluate_scores()
        self._identify_uncovered()
        self._collect_keywords()

    def _format_percentage(self, part: int, whole: int) -> str:
        if whole == 0:
            return "0.00%"
        return f"{(part / whole) * 100:.2f}%"

    def _build_top_n_table(self) -> pd.DataFrame:
        columns = [
            "question",
            "evaluated_score",
            "evaluated_topic",
            "evaluated_kb_question",
        ]
        available_columns = [col for col in columns if col in self.uncovered_df.columns]
        return (
            self.uncovered_df.sort_values("evaluated_score", na_position="first")
            .head(self.top_n)[available_columns]
            .rename(
                columns={
                    "evaluated_score": "top_score",
                    "evaluated_topic": "topic",
                    "evaluated_kb_question": "kb_question",
                }
            )
        )

    def _topic_distribution(self) -> pd.Series:
        if "evaluated_topic" not in self.uncovered_df:
            return pd.Series(dtype=int)
        series = self.uncovered_df["evaluated_topic"].dropna()
        if series.empty:
            return pd.Series(dtype=int)
        return series.value_counts().sort_values(ascending=False)

    def _top_keywords(self, limit: int = 10) -> List[Tuple[str, int]]:
        return self.keyword_counter.most_common(limit)

    def report(self) -> str:
        total = len(self.df)
        uncovered = len(self.uncovered_df)
        coverage = total - uncovered
        coverage_pct = self._format_percentage(coverage, total)
        uncovered_pct = self._format_percentage(uncovered, total)

        lines = [
            "=== Показатели покрытия ===",
            f"Всего запросов: {total}",
            f"Покрыты знаниями: {coverage} ({coverage_pct})",
            f"Не покрыты: {uncovered} ({uncovered_pct})",
            f"Порог релевантности: {self.threshold}",
        ]

        lines.append("\n=== Проблемные вопросы ===")
        top_table = self._build_top_n_table()
        lines.append(top_table.to_string(index=False) if not top_table.empty else "(нет проблемных вопросов)")

        lines.append("\n=== Распределение по темам ===")
        topic_dist = self._topic_distribution()
        lines.append(topic_dist.to_string() if not topic_dist.empty else "(темы не определены)")

        lines.append("\n=== Ключевые слова ===")
        keywords = self._top_keywords(limit=15)
        if keywords:
            lines.extend([f"{word}: {count}" for word, count in keywords])
        else:
            lines.append("(ключевые слова не обнаружены)")

        return "\n".join(lines)

    def export(self, out_path: Path) -> None:
        top_table = self._build_top_n_table()
        if out_path.suffix.lower() == ".csv":
            top_table.to_csv(out_path, index=False)
        elif out_path.suffix.lower() in {".md", ".markdown"}:
            content = ["| Вопрос | Оценка | Тема |", "| --- | --- | --- |"]
            for _, row in top_table.iterrows():
                question = str(row.get("question", "")).replace("|", "\\|")
                score = row.get("top_score", "")
                topic = str(row.get("topic", "")).replace("|", "\\|")
                content.append(f"| {question} | {score} | {topic} |")
            out_path.write_text("\n".join(content), encoding="utf-8")
        else:
            top_table.to_csv(out_path, index=False)


def tokenize(text: str) -> List[str]:
    if not text:
        return []
    tokens = TOKEN_PATTERN.findall(text.lower())
    return [token for token in tokens if token not in STOP_WORDS and not token.isdigit() and len(token) > 1]


def _upload_report_to_drive(
    report_text: str,
    drive_client: Optional[GoogleDriveClient],
    folder_id: Optional[str],
    *,
    file_id: Optional[str] = None,
) -> None:
    if not drive_client or not folder_id or not drive_client.is_configured:
        return

    tmp_path: Optional[Path] = None

    with NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(report_text)
        tmp_path = Path(tmp_file.name)

    try:
        if tmp_path:
            drive_client.upload_or_update_file(
                tmp_path,
                folder_id,
                file_name="coverage_report.txt",
                mime_type="text/plain",
                file_id=file_id,
                file_id_env_var="GOOGLE_DRIVE_REPORTS_FILE_ID",
            )
    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coverage of LegalBot knowledge base using logs")
    default_log = Path("data/log.csv")
    default_knowledge = Path("data/knowledge.csv")

    parser.add_argument("--log", type=Path, default=default_log, help="Path to CSV log file")
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=default_knowledge,
        help="Path to knowledge CSV used for score recomputation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=60.0,
        help="Relevance threshold below which questions are considered uncovered",
    )
    parser.add_argument("--top-n", type=int, default=10, help="Number of problematic questions to display")

    recompute_group = parser.add_mutually_exclusive_group()
    recompute_group.add_argument(
        "--use-stored-score",
        dest="use_stored_score",
        action="store_true",
        help="Use stored top_score from logs (default)",
    )
    recompute_group.add_argument(
        "--recompute-score",
        dest="use_stored_score",
        action="store_false",
        help="Recompute score/topic using the knowledge base",
    )
    parser.set_defaults(use_stored_score=True)

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to export the table of problematic questions (CSV or Markdown)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log_path = args.log.expanduser()
    knowledge_path = args.knowledge.expanduser()

    if not log_path.is_absolute():
        log_path = (Path.cwd() / log_path).resolve()
    if not knowledge_path.is_absolute():
        knowledge_path = (Path.cwd() / knowledge_path).resolve()

    evaluator = CoverageEvaluator(
        log_path=log_path,
        knowledge_path=knowledge_path,
        threshold=args.threshold,
        use_stored_score=args.use_stored_score,
        top_n=args.top_n,
    )

    evaluator.load()
    evaluator.evaluate()

    report_text = evaluator.report()
    print(report_text)

    config = Config.load(allow_missing=True)
    drive_client = (
        GoogleDriveClient(config.google_drive_credentials_file)
        if config.google_drive_credentials_file
        else None
    )

    _upload_report_to_drive(
        report_text,
        drive_client,
        config.google_drive_reports_folder_id,
        file_id=config.google_drive_reports_file_id,
    )

    if args.out:
        out_path = args.out.expanduser()
        if not out_path.is_absolute():
            out_path = (Path.cwd() / out_path).resolve()
        evaluator.export(out_path)
        print(f"\n[i] Результаты сохранены в {out_path}")

        if (
            drive_client
            and drive_client.is_configured
            and config.google_drive_reports_folder_id
        ):
            drive_client.upload_or_update_file(
                out_path,
                config.google_drive_reports_folder_id,
                file_id=config.google_drive_reports_file_id,
                file_id_env_var="GOOGLE_DRIVE_REPORTS_FILE_ID",
            )


if __name__ == "__main__":
    main()
