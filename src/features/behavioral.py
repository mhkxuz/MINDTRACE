"""
Behavioral interaction feature extraction module.
Faithfully reproduces the behavioral feature logic from:
notebooks/data_preparation+behavioral_features.ipynb
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import stats

# Exact 8 features used in fusion_model.ipynb
FUSION_BEHAVIORAL_COLS = [
    "total_turn_count",
    "engagement_ratio",
    "avg_user_msg_length",
    "msg_length_slope",
    "dropout_flag",
    "dropout_gap",
    "user_msg_length_std",
    "ended_by_user",
]

# All 13 computed behavioral metrics
ALL_BEHAVIORAL_METRICS = [
    "user_turn_count",
    "assistant_turn_count",
    "total_turn_count",
    "engagement_ratio",
    "avg_user_msg_length",
    "avg_assistant_msg_length",
    "user_msg_length_std",
    "last_user_turn",
    "last_speaker",
    "ended_by_user",
    "msg_length_slope",
    "dropout_gap",
    "dropout_flag",
]


def clean_text(text: Any) -> str:
    """Normalize whitespace and strip text."""
    if text is None or pd.isna(text):
        return ""
    text_str = str(text)
    text_str = re.sub(r"\s+", " ", text_str)
    return text_str.strip()


def parse_conversation_transcript(transcript_text: str) -> List[Dict[str, str]]:
    """
    Parse a plain text transcript into structured turns.
    Supports formats like:
      User: Hello, I feel overwhelmed.
      Assistant: I am here to listen.
      or
      Seeker: ...
      Supporter: ...
    """
    lines = transcript_text.strip().split("\n")
    turns = []
    current_speaker = "user"

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for speaker prefixes
        match = re.match(r"^(user|seeker|client|patient|person|me)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match:
            current_speaker = "user"
            msg = clean_text(match.group(2))
            if msg:
                turns.append({"speaker": "user", "message": msg})
            continue

        match_asst = re.match(r"^(assistant|supporter|bot|counselor|therapist|system|sys)\s*:\s*(.*)$", line, re.IGNORECASE)
        if match_asst:
            current_speaker = "assistant"
            msg = clean_text(match_asst.group(2))
            if msg:
                turns.append({"speaker": "assistant", "message": msg})
            continue

        # If line has no prefix, treat it as continuing the current speaker's turn or alternating
        turns.append({"speaker": current_speaker, "message": clean_text(line)})

    return turns


def extract_behavioral_features(turns: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract all 13 behavioral features from a list of turn dicts:
    [{'speaker': 'user'|'assistant', 'message': str}, ...]
    """
    if not turns:
        return {
            "user_turn_count": 0,
            "assistant_turn_count": 0,
            "total_turn_count": 0,
            "engagement_ratio": 0.0,
            "avg_user_msg_length": 0.0,
            "avg_assistant_msg_length": 0.0,
            "user_msg_length_std": 0.0,
            "last_user_turn": 0,
            "last_speaker": "none",
            "ended_by_user": 0,
            "msg_length_slope": 0.0,
            "dropout_gap": 0,
            "dropout_flag": 0,
            "turns_data": [],
        }

    valid_turns = []
    for idx, t in enumerate(turns, start=1):
        spk = str(t.get("speaker", "")).strip().lower()
        if spk in ["user", "seeker", "usr", "client", "patient", "me"]:
            speaker_norm = "user"
        elif spk in ["assistant", "supporter", "sys", "system", "bot", "counselor", "therapist"]:
            speaker_norm = "assistant"
        else:
            speaker_norm = "user"  # default fallback

        msg = clean_text(t.get("message", ""))
        words = len(msg.split()) if msg else 0
        valid_turns.append({
            "turn_number": idx,
            "speaker": speaker_norm,
            "message": msg,
            "word_count": words,
            "char_count": len(msg),
        })

    total_turn_count = len(valid_turns)
    user_turns = [t for t in valid_turns if t["speaker"] == "user"]
    asst_turns = [t for t in valid_turns if t["speaker"] == "assistant"]

    user_turn_count = len(user_turns)
    assistant_turn_count = len(asst_turns)

    engagement_ratio = (user_turn_count / total_turn_count) if total_turn_count > 0 else 0.0

    user_lengths = [t["word_count"] for t in user_turns]
    asst_lengths = [t["word_count"] for t in asst_turns]

    avg_user_msg_length = float(np.mean(user_lengths)) if user_lengths else 0.0
    avg_assistant_msg_length = float(np.mean(asst_lengths)) if asst_lengths else 0.0
    user_msg_length_std = float(np.std(user_lengths, ddof=1)) if len(user_lengths) > 1 else 0.0

    last_user_turn = max([t["turn_number"] for t in user_turns]) if user_turns else 0
    last_speaker = valid_turns[-1]["speaker"] if valid_turns else "none"
    ended_by_user = 1 if last_speaker == "user" else 0

    # msg_length_slope: Linear regression slope over user turns
    if len(user_lengths) >= 2:
        slope, _, _, _, _ = stats.linregress(range(len(user_lengths)), user_lengths)
        msg_length_slope = float(slope)
    else:
        msg_length_slope = 0.0

    # dropout_gap = total_turn_count - last_user_turn
    dropout_gap = total_turn_count - last_user_turn
    dropout_flag = 1 if dropout_gap > 2 else 0

    return {
        "user_turn_count": user_turn_count,
        "assistant_turn_count": assistant_turn_count,
        "total_turn_count": total_turn_count,
        "engagement_ratio": float(engagement_ratio),
        "avg_user_msg_length": float(avg_user_msg_length),
        "avg_assistant_msg_length": float(avg_assistant_msg_length),
        "user_msg_length_std": float(user_msg_length_std),
        "last_user_turn": last_user_turn,
        "last_speaker": last_speaker,
        "ended_by_user": ended_by_user,
        "msg_length_slope": float(msg_length_slope),
        "dropout_gap": dropout_gap,
        "dropout_flag": dropout_flag,
        "turns_data": valid_turns,
    }


def get_fusion_feature_vector(features: Dict[str, Any]) -> np.ndarray:
    """
    Extract the 8 features in the exact order required by the fusion model:
    ['total_turn_count', 'engagement_ratio', 'avg_user_msg_length',
     'msg_length_slope', 'dropout_flag', 'dropout_gap',
     'user_msg_length_std', 'ended_by_user']
    Returns a 1D numpy array of shape (8,).
    """
    vector = [float(features.get(col, 0.0)) for col in FUSION_BEHAVIORAL_COLS]
    return np.array(vector, dtype=np.float32)


def process_conversation_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Process an uploaded dialogue turns dataframe.
    Expected columns: either 'conversation_id', 'turn_number', 'speaker', 'message'
    or simple 'speaker', 'message'.
    """
    if "conversation_id" not in df.columns:
        df["conversation_id"] = 1

    records = []
    for conv_id, group in df.groupby("conversation_id"):
        group = group.sort_values(by="turn_number") if "turn_number" in group.columns else group
        turns = []
        for _, row in group.iterrows():
            turns.append({
                "speaker": row.get("speaker", "user"),
                "message": row.get("message", row.get("text", "")),
            })
        feats = extract_behavioral_features(turns)
        feats["conversation_id"] = conv_id
        records.append(feats)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if k != "turns_data"} for r in records])
    return summary_df, records
