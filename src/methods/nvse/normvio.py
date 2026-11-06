"""NormVio model loading and post-category scoring for NVSE."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List
import unicodedata

from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import BertModel, BertTokenizer

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.layout import ITEM_NVSE_COLUMNS, REDDIT, load_items  # noqa: E402
from src.utils.splits import discover_splits  # noqa: E402

# Constants
BERT_TYPE = "DeepPavlov/bert-base-cased-conversational"
HIDDEN_SIZE = 768
MAX_LENGTH = 120
DROPOUT = 0.1
NUM_CLASSES = 2

CATEGORIES = [
    "spam",
    "meta-rules",
    "content",
    "harassment",
    "hatespeech",
    "format",
    "off-topic",
    "trolling",
    "incivility",
]

THRESHOLD_GRID = np.linspace(-1.5, 1.5, 300)
MIN_VOTES_SKILL = 5  # min votes per userxcommunityxcategory to compute skill


# Model definitions from NormVio


class EncoderBERT(nn.Module):
    """Implement encoder bert."""

    def __init__(self, device: torch.device, model_source: str | Path | None = None):
        """Initialize the instance."""
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained(str(model_source or resolve_bert_source())).to(
            device
        )

    def forward(self, input_seq, input_lengths):
        """Run a forward pass through the model."""
        input_seq = input_seq.T  # Convert sequence-first to batch-first.
        mask_ids = (input_seq != 0).long()
        token_ids = torch.ones_like(input_seq)
        return self.model(input_ids=input_seq, attention_mask=mask_ids, token_type_ids=token_ids)[
            1
        ]  # pooled CLS


class SingleTargetClf(nn.Module):
    """Implement single target clf."""

    def __init__(self, hidden_size: int, num_classes: int, dropout: float, device: torch.device):
        """Initialize the instance."""
        super().__init__()
        self.device = device
        self.layer1 = nn.Linear(hidden_size, hidden_size).to(device)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2).to(device)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, num_classes).to(device)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, ctx_out, dialog_lengths):
        """Run a forward pass through the model."""
        # ctx_out: [1, batch, hidden]
        lengths = (
            dialog_lengths.unsqueeze(0).unsqueeze(2).expand(1, -1, ctx_out.size(2)).to(self.device)
        )
        last = torch.gather(ctx_out, 0, lengths - 1).squeeze(0)  # [batch, hidden]
        if last.dim() == 1:
            last = last.unsqueeze(0)
        h = self.layer2_act(self.layer2(self.drop(self.layer1_act(self.layer1(self.drop(last))))))
        return self.clf(self.drop(h))


class Predictor(nn.Module):
    """Implement predictor."""

    def __init__(self, encoder: EncoderBERT, clf: SingleTargetClf):
        """Initialize the instance."""
        super().__init__()
        self.encoder = encoder
        self.clf = clf

    def forward(self, input_batch, dialog_lengths, utt_lengths):
        """Run a forward pass through the model."""
        utt_hidden = self.encoder(input_batch, utt_lengths)  # [batch, hidden]
        ctx_out = utt_hidden.unsqueeze(0)  # [1, batch, hidden]
        return self.clf(ctx_out, dialog_lengths)


# Utilities


def get_device(device_str: str) -> torch.device:
    """Return device for the supplied input."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def unicode_to_ascii(s: str) -> str:
    """Normalize text to ASCII characters."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def encode_text(text: str, tokenizer: BertTokenizer) -> List[int]:
    """Tokenize and encode a batch of texts."""
    cleaned = unicode_to_ascii(text.strip())
    if not cleaned:
        return []
    return tokenizer.encode(
        cleaned, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH
    )


def resolve_bert_source(model_id: str = BERT_TYPE) -> Path | str:
    """Prefer the exact locally cached Hugging Face snapshot when available."""
    try:
        return Path(snapshot_download(model_id, local_files_only=True))
    except FileNotFoundError:
        return model_id


def load_model(
    model_dir: Path,
    device: torch.device,
    model_source: str | Path | None = None,
) -> Predictor:
    """Load model from its configured source."""
    pt_path = model_dir / "finetuned_model.pt"
    # Legacy checkpoints use pickle serialization.
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    enc = EncoderBERT(device, model_source=model_source)
    enc.load_state_dict(ckpt["en"])
    enc.eval()
    clf = SingleTargetClf(HIDDEN_SIZE, NUM_CLASSES, DROPOUT, device)
    clf.load_state_dict(ckpt["atk_clf"])
    clf.eval()
    return Predictor(enc, clf).to(device)


@torch.no_grad()
def score_texts(
    texts: List[str],
    predictor: Predictor,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int = 32,
    encoded_texts: List[List[int]] | None = None,
) -> np.ndarray:
    """Predict violation probabilities for a batch of texts.

    ``encoded_texts`` (``encode_text`` applied to each text) lets callers that
    score the same texts with several models tokenize them only once.
    """
    probs = np.full(len(texts), np.nan)
    valid_idx = [i for i, t in enumerate(texts) if t.strip()]

    n_batches = (len(valid_idx) + batch_size - 1) // batch_size
    for batch_num, start in enumerate(range(0, len(valid_idx), batch_size)):
        if batch_num == 0:
            print(f"Inference: {len(valid_idx):,} texts | {n_batches} batches", flush=True)
        chunk = valid_idx[start : start + batch_size]
        batch_texts = [texts[i] for i in chunk]
        encoded = (
            [encoded_texts[i] for i in chunk]
            if encoded_texts is not None
            else [encode_text(t, tokenizer) for t in batch_texts]
        )

        nonempty = [(li, e) for li, e in enumerate(encoded) if e]
        if not nonempty:
            continue
        local_idx, enc_clean = zip(*nonempty)

        max_len = max(len(e) for e in enc_clean)
        padded = [list(e) + [0] * (max_len - len(e)) for e in enc_clean]

        inp = torch.LongTensor(padded).T.to(device)
        lengths = torch.tensor([len(e) for e in enc_clean], dtype=torch.long).to(device)
        d_lens = torch.ones(len(enc_clean), dtype=torch.long).to(device)

        logits = predictor(inp, d_lens, lengths)
        p = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        for li, pi in zip(local_idx, p):
            probs[chunk[li]] = float(pi)

    return probs


def run_inference_all_categories(
    texts: List[str],
    models_dir: Path,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int,
    model_source: str | Path | None = None,
) -> Dict[str, np.ndarray]:
    """Run the inference all categories workflow."""
    scores: Dict[str, np.ndarray] = {}
    model_source = model_source or resolve_bert_source()
    print(f"BERT source: {model_source}")
    # Every category model shares the tokenizer: tokenize once, not per model.
    encoded_texts = [encode_text(text, tokenizer) if text.strip() else [] for text in texts]
    for cat in tqdm(CATEGORIES, desc="NormVio categories"):
        model_dir = models_dir / cat
        if not model_dir.exists():
            print(f"Skipped {cat}: model missing at {model_dir}")
            scores[cat] = np.full(len(texts), np.nan)
            continue
        try:
            predictor = load_model(model_dir, device, model_source=model_source)
            probs = score_texts(
                texts, predictor, tokenizer, device, batch_size, encoded_texts=encoded_texts
            )
            scores[cat] = probs
        except Exception as exc:
            import traceback

            print(f"Failed {cat}: {exc}")
            traceback.print_exc()
            scores[cat] = np.full(len(texts), np.nan)
        finally:
            try:
                del predictor
            except NameError:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return scores


def load_votes(votes_dir: Path) -> pd.DataFrame:
    """Load the union of the explicitly configured prepared splits."""
    splits = discover_splits(votes_dir)
    if not splits:
        raise FileNotFoundError(f"No complete prepared split found under {votes_dir}")
    frames = []
    for split_path in splits.values():
        for partition in ("train", "val", "test"):
            frames.append(pd.read_parquet(split_path / f"{partition}_votes.parquet"))
    votes = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["username", "item_id"], keep="first"
    )
    print(
        f"Votes: {len(votes):,} rows | {votes['item_id'].nunique():,} posts | source={votes_dir}"
    )
    votes["vote"] = votes["vote"].astype(float)
    # Labels are +1 for approve and -1 for remove.
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    required = ["username", "item_id", "timestamp", "vote", "community", "label"]
    missing = set(required) - set(votes.columns)
    if missing:
        raise ValueError(
            f"Votes are missing columns required for causal metadata: {sorted(missing)}"
        )
    return votes[required]


def _load_split_votes(path: Path) -> pd.DataFrame:
    """Load split votes from its configured source."""
    votes = pd.read_parquet(path)
    votes["vote"] = votes["vote"].astype(float)
    if "label" not in votes.columns:
        votes["label"] = float("nan")
    else:
        votes["label"] = votes["label"].astype(float)
    required = ["username", "item_id", "timestamp", "vote", "community", "label"]
    missing = set(required) - set(votes.columns)
    if missing:
        raise ValueError(f"Split votes are missing required columns: {sorted(missing)}")
    return votes[required]


# Post violation scoring


def _build_post_text(row: pd.Series) -> str:
    """Build post text from the supplied data."""

    def clean_text(value: object) -> str:
        return "" if value is None or pd.isna(value) else str(value).strip()

    subreddit = clean_text(row.get("community", row.get("subreddit", "")))
    if subreddit[:2].casefold() == "r/":
        subreddit = subreddit[2:]
    title = clean_text(row.get("title", ""))
    # Canonical rows use raw ``selftext``; legacy rows use combined ``text``.
    body_field = "selftext" if "selftext" in row.index else "text"
    body = clean_text(row.get(body_field, ""))
    body = "" if body in ("[removed]", "[deleted]") else body
    content = (title + " " + body).strip()
    return f"r/{subreddit} {content}" if content else ""


POST_TEXTS_PATH = REDDIT.default_inputs().post_texts
DEFAULT_VIOLATION_SCORES = REDDIT.nvse_scores


def score_violations(
    votes_dir: Path,
    models_dir: Path,
    out_path: Path,
    tokenizer: BertTokenizer,
    device: torch.device,
    batch_size: int,
    max_posts: int = None,
    post_texts_path: Path = POST_TEXTS_PATH,
    model_source: str | Path | None = None,
) -> pd.DataFrame:
    """Score every post with the available violation models."""
    votes = load_votes(votes_dir)
    items = votes.drop_duplicates("item_id")[["item_id", "community"]].copy()

    # Item tables provide title/selftext (derived from ``text`` when absent).
    post_texts = load_items(post_texts_path, required=ITEM_NVSE_COLUMNS)
    posts = items.merge(
        post_texts[list(ITEM_NVSE_COLUMNS)].drop_duplicates("item_id"), on="item_id", how="left"
    )
    n_missing = posts["title"].isna().sum()
    source = post_texts_path

    print(f"Post texts: {len(posts):,} | missing={n_missing:,} | source={source}")

    if max_posts is not None:
        posts = posts.head(max_posts)
        print(f"Post limit: {max_posts:,}")
    posts["input_text"] = posts.apply(_build_post_text, axis=1)
    texts = posts["input_text"].tolist()

    scores = run_inference_all_categories(
        texts,
        models_dir,
        tokenizer,
        device,
        batch_size,
        model_source=model_source,
    )

    for cat, arr in scores.items():
        posts[f"score_{cat}"] = arr

    score_mat = np.stack([scores[c] for c in CATEGORIES], axis=1)
    # All-NaN rows indicate that no category model loaded.
    all_nan_rows = np.all(np.isnan(score_mat), axis=1)
    top_cats, top_scores = [], []
    for i, row in enumerate(score_mat):
        if all_nan_rows[i]:
            top_cats.append(None)
            top_scores.append(np.nan)
        else:
            top_cats.append(CATEGORIES[int(np.nanargmax(row))])
            top_scores.append(float(np.nanmax(row)))
    posts["top_violation_category"] = top_cats
    posts["top_violation_score"] = top_scores
    if all_nan_rows.all():
        print("No violation model produced scores; check PyTorch and model files")

    keep = ["item_id", "community", "top_violation_category", "top_violation_score"] + [
        f"score_{c}" for c in CATEGORIES
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    posts[keep].to_parquet(out_path, index=False)
    n_scored = posts["top_violation_category"].notna().sum()
    print(f"Violation scores: {out_path} | scored={n_scored:,}/{len(posts):,}")
    return posts[keep]
