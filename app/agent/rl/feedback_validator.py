import re
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from catboost import CatBoostClassifier

POS = set("хорошо отлично полезно верно корректно понятно ясно good great helpful valid precise clear informative".split())
NEG = set("плохо неверно бесполезно спам ужасно ошибка баг мусор непонятно ерунда бред говно wrong bad useless unclear misleading".split())
PROFANITY_ROOTS = [
    "хуй", "хуйн", "бля", "блять", "сука", "пидор", "муд", "хер", "жоп", "идиот", "дебил",
    "fuck", "shit", "bitch", "asshole", "bastard", "crap", "dick", "pussy", "cunt"
]


def norm(s):
    return re.sub(r'\s+', ' ', str(s).strip()) if pd.notna(s) else ""


def token_set(t):
    return set(re.findall(r"[A-Za-zА-Яа-яЁё0-9']+", str(t).lower()))


def jacc(a, b):
    if not a and not b: return 0.0
    return len(a & b) / (len(a | b) + 1e-9)


def senti(t):
    toks = re.findall(r"[A-Za-zА-Яа-яЁё0-9']+", str(t).lower())
    return sum(1 for x in toks if x in POS) - sum(1 for x in toks if x in NEG)


def tox(t):
    s = str(t).lower()
    return int(any(root in s for root in PROFANITY_ROOTS))


def react_cons(r, s):
    r = str(r).lower().strip()
    if r in ["like", "liked", "1", "true", "👍", "лайк", "нравится"]:
        rb = 1
    elif r in ["dislike", "disliked", "0", "false", "👎", "дизлайк", "не нравится"]:
        rb = 0
    else:
        return 0
    return 1 if (s >= 0 and rb == 1) or (s < 0 and rb == 0) else -1


# --- The Validator Class ---
class FeedbackValidator:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = self._load_model(model_path)
        self.tfidf: TfidfVectorizer = joblib.load(vectorizer_path)

    def _load_model(self, model_path: str):
        if model_path.endswith(".cbm") and CatBoostClassifier:
            model = CatBoostClassifier()
            model.load_model(model_path)
            return model
        # Isolation Forest model
        elif model_path.endswith(".joblib"):
            return joblib.load(model_path)
        else:
            raise ValueError(f"Unsupported model format or missing library for: {model_path}")

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["answer_norm"] = df["answer"].astype(str).map(norm)
        df["comment_norm"] = df["comment"].astype(str).map(norm)

        stacked = pd.concat([df["answer_norm"], df["comment_norm"]], axis=0)
        tfidf_matrix = self.tfidf.transform(stacked)
        A, C = tfidf_matrix[:len(df)], tfidf_matrix[len(df):]
        cos_sim = cosine_similarity(A, C).diagonal()

        tokA = df["answer_norm"].map(token_set)
        tokC = df["comment_norm"].map(token_set)
        jacc_overlap = [jacc(a, c) for a, c in zip(tokA, tokC)]

        c_len_chars = df["comment_norm"].map(len).values
        c_len_words = df["comment_norm"].map(lambda t: len(t.split())).values
        sent_c = df["comment_norm"].map(senti).values
        tox_c = df["comment_norm"].map(tox).values
        consistency = [react_cons(r, s) for r, s in zip(df["reaction"], sent_c)]

        feats = pd.DataFrame({
            "cosine_sim": cos_sim,
            "jaccard_overlap": jacc_overlap,
            "c_len_chars": c_len_chars,
            "c_len_words": c_len_words,
            "sentiment_comment": sent_c,
            "toxicity_comment": tox_c,
            "reaction_consistency": consistency,
        }).fillna(0)
        return feats

    def is_valid(self, feedback_data: dict, prob_threshold: float = 0.5) -> bool:
        """
        Predicts if a single feedback entry is valid.

        Args:
            feedback_data: A dictionary with keys 'answer', 'comment', and 'reaction'.
            prob_threshold: The probability threshold to classify as valid.

        Returns:
            True if the feedback is predicted to be valid, False otherwise.
        """
        df = pd.DataFrame([feedback_data])

        df.rename(columns={
            'answer': 'answer',
            'comment': 'comment',
            'rating': 'reaction'  # Assuming rating maps to reaction
        }, inplace=True)

        features = self._build_features(df)

        # Ensure feature order matches the model's training order
        if hasattr(self.model, 'feature_names_') and self.model.feature_names_ is not None:
            features = features[self.model.feature_names_]

        prediction_prob = self.model.predict_proba(features.values)[:, 1]

        return prediction_prob[0] >= prob_threshold
