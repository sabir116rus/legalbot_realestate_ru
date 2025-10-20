
import os
import pandas as pd
from rapidfuzz import process, fuzz

class KnowledgeBase:
    def __init__(self, csv_path: str):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Knowledge CSV not found: {csv_path}")
        self.df = pd.read_csv(csv_path, encoding="utf-8-sig")

        def _normalize_column(name):
            if not isinstance(name, str):
                return name
            # Удаляем невидимые символы (например, BOM) и пробелы по краям
            return name.strip().lstrip("\ufeff")

        self.df.rename(columns=_normalize_column, inplace=True)

        if "\ufeffid" in self.df.columns:
            self.df.rename(columns={"\ufeffid": "id"}, inplace=True)

        if "id" not in self.df.columns:
            raise ValueError("Knowledge CSV must contain an 'id' column after normalization")

        if self.df["id"].isnull().any():
            raise ValueError("Knowledge CSV contains empty values in the 'id' column")

        # Текст для поиска: объединяем поля вопрос+ответ+тема
        self.df["search_text"] = (
            self.df["topic"].fillna("").astype(str) + " | " +
            self.df["question"].fillna("").astype(str) + " | " +
            self.df["answer"].fillna("").astype(str)
        )
        self.corpus = self.df["search_text"].tolist()

    def query(self, user_question: str, top_k: int = 3):
        if not user_question or not user_question.strip():
            return []

        # RapidFuzz: быстрый нечеткий поиск по корпусу
        matches = process.extract(
            query=user_question,
            choices=self.corpus,
            scorer=fuzz.WRatio,
            limit=top_k
        )

        # matches: list of tuples (text, score, index)
        results = []
        for _, score, idx in matches:
            row = self.df.iloc[idx].to_dict()
            row["score"] = int(score)
            results.append(row)
        return results

def build_context_snippets(rows):
    """Формирует текстовый контекст для подсказки модели."""
    parts = []
    for r in rows:
        row_id = r.get("id")
        row_id = "" if pd.isna(row_id) else str(row_id)
        part = (
            f"[ID:{row_id}] Тема: {r.get('topic')}\n"
            f"Вопрос: {r.get('question')}\n"
            f"Ответ: {r.get('answer')}\n"
            f"Правовые ссылки: {r.get('law_refs')}\n"
            f"Источник: {r.get('url')}\n"
            f"(релевантность: {r.get('score')})\n"
        )
        parts.append(part)
    return "\n---\n".join(parts)
