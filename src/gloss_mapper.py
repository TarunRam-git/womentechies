import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


ARTICLES = {"a", "an", "the"}
AUX_WORDS = {
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "has", "have", "had", "will", "would", "shall", "should", "can", "could", "may", "might"
}
NEGATIONS = {"not", "never", "no", "dont", "don't", "cannot", "can't"}
PRONOUNS = {"i", "you", "he", "she", "it", "we", "they"}
CUSTOM_SENTENCE_GLOSSES = {
    "i need help": ["I", "HELP"],
    "need help": ["HELP"],
    "help me": ["HELP", "I"],
}


@dataclass
class GlossResult:
    input_text: str
    glosses: List[str]
    oov_words: List[str]


class GlossMapper:
    def __init__(self, gloss_dict_path: Path, oov_log_path: Path):
        self.gloss_dict = self._load_dict(gloss_dict_path)
        self.oov_log_path = oov_log_path
        self.oov_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_dict(self, path: Path) -> Dict[str, str]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {k.lower(): v.upper() for k, v in data.items()}

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

    def _apply_rules(self, tokens: List[str]) -> Tuple[List[str], bool]:
        neg = any(tok in NEGATIONS for tok in tokens)
        filtered = [
            tok for tok in tokens
            if tok not in ARTICLES and tok not in AUX_WORDS and tok not in NEGATIONS
        ]

        subjects = [t for t in filtered if t in PRONOUNS]
        others = [t for t in filtered if t not in PRONOUNS]

        verbs = []
        non_verbs = []
        for tok in others:
            if tok.endswith("ing") or tok.endswith("ed"):
                verbs.append(tok)
            else:
                non_verbs.append(tok)

        # Approximate ISL style SOV: subject + object + verb
        ordered = subjects + non_verbs + verbs
        return ordered, neg

    def _lookup_gloss(self, token: str) -> str:
        if token in self.gloss_dict:
            return self.gloss_dict[token]
        title = token.replace("_", " ").upper()
        if title in self.gloss_dict.values():
            return title
        return ""

    def _fingerspell(self, token: str) -> List[str]:
        chars = [c for c in token.upper() if c.isalnum()]
        return [f"FS_{c}" for c in chars]

    def _map_phrases(self, tokens: List[str], max_ngram: int = 5) -> Tuple[List[str], List[str]]:
        glosses: List[str] = []
        residual: List[str] = []

        i = 0
        while i < len(tokens):
            matched = False
            upper_n = min(max_ngram, len(tokens) - i)
            for n in range(upper_n, 1, -1):
                phrase = " ".join(tokens[i:i + n])
                gloss = self._lookup_gloss(phrase)
                if gloss:
                    glosses.append(gloss)
                    i += n
                    matched = True
                    break

            if matched:
                continue

            residual.append(tokens[i])
            i += 1

        return glosses, residual

    def _log_oov(self, oov: List[str]):
        if not oov:
            return
        with self.oov_log_path.open("a", encoding="utf-8") as f:
            for word in oov:
                f.write(word + "\n")

    def map_text(self, text: str) -> GlossResult:
        normalized_text = " ".join(self._tokenize(text))
        if normalized_text in CUSTOM_SENTENCE_GLOSSES:
            glosses = CUSTOM_SENTENCE_GLOSSES[normalized_text]
            return GlossResult(input_text=text, glosses=glosses, oov_words=[])

        tokens = self._tokenize(text)
        phrase_glosses, residual_tokens = self._map_phrases(tokens)
        ordered, neg = self._apply_rules(residual_tokens)

        glosses = []
        oov = []

        glosses.extend(phrase_glosses)
        for tok in ordered:
            gloss = self._lookup_gloss(tok)
            if gloss:
                glosses.append(gloss)
            else:
                oov.append(tok)
                glosses.extend(self._fingerspell(tok))

        if neg:
            glosses.append("NOT")

        self._log_oov(oov)
        return GlossResult(input_text=text, glosses=glosses, oov_words=oov)
