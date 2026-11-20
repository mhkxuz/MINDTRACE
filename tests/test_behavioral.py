"""
Unit tests for behavioral interaction feature extraction.
"""

import numpy as np
import pandas as pd
from src.features.behavioral import (
    clean_text,
    parse_conversation_transcript,
    extract_behavioral_features,
    get_fusion_feature_vector,
    process_conversation_dataframe,
    FUSION_BEHAVIORAL_COLS,
)


def test_clean_text():
    assert clean_text("  hello   world  \n ") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_parse_conversation_transcript():
    transcript = """
    User: I feel down.
    Assistant: I am here to help.
    User: Thank you.
    """
    turns = parse_conversation_transcript(transcript)
    assert len(turns) == 3
    assert turns[0]["speaker"] == "user"
    assert turns[0]["message"] == "I feel down."
    assert turns[1]["speaker"] == "assistant"
    assert turns[2]["speaker"] == "user"


def test_empty_turns():
    feats = extract_behavioral_features([])
    assert feats["total_turn_count"] == 0
    assert feats["user_turn_count"] == 0
    assert feats["assistant_turn_count"] == 0
    assert feats["engagement_ratio"] == 0.0
    assert feats["msg_length_slope"] == 0.0

    vector = get_fusion_feature_vector(feats)
    assert vector.shape == (8,)
    assert np.all(vector == 0.0)


def test_malformed_and_whitespace_transcripts():
    """Test handling of None, pure whitespace, and non-standard input."""
    # Empty string
    turns_empty = parse_conversation_transcript("")
    assert len(turns_empty) == 0

    # Whitespace only
    turns_ws = parse_conversation_transcript("   \n\n\t  \n  ")
    assert len(turns_ws) == 0

    # Unrecognized speaker tags (should default gracefully to user)
    turns_custom = parse_conversation_transcript("Patient: I am anxious.\nCounselor: Tell me.")
    assert len(turns_custom) == 2
    assert turns_custom[0]["speaker"] == "user"
    assert turns_custom[1]["speaker"] == "assistant"

    # Lines without speaker prefix
    turns_raw = parse_conversation_transcript("Just a rambling sentence with no speaker prefix.")
    assert len(turns_raw) == 1
    assert turns_raw[0]["speaker"] == "user"


def test_all_assistant_turns_no_user():
    """Conversation where only assistant speaks."""
    turns = [
        {"speaker": "assistant", "message": "Are you there?"},
        {"speaker": "assistant", "message": "Please reply."},
    ]
    feats = extract_behavioral_features(turns)
    assert feats["total_turn_count"] == 2
    assert feats["user_turn_count"] == 0
    assert feats["assistant_turn_count"] == 2
    assert feats["engagement_ratio"] == 0.0
    assert feats["avg_user_msg_length"] == 0.0
    assert feats["last_speaker"] == "assistant"
    assert feats["ended_by_user"] == 0
    assert feats["dropout_flag"] == 0


def test_single_turn():
    turns = [{"speaker": "user", "message": "Help me please."}]
    feats = extract_behavioral_features(turns)
    assert feats["total_turn_count"] == 1
    assert feats["user_turn_count"] == 1
    assert feats["assistant_turn_count"] == 0
    assert feats["engagement_ratio"] == 1.0
    assert feats["last_speaker"] == "user"
    assert feats["ended_by_user"] == 1
    assert feats["msg_length_slope"] == 0.0
    assert feats["dropout_gap"] == 0
    assert feats["dropout_flag"] == 0


def test_high_distress_withdrawal_dialogue():
    """
    Test a conversation where user shrinks message length and leaves assistant hanging.
    Turn 1 (user): 10 words
    Turn 2 (asst): 5 words
    Turn 3 (user): 5 words
    Turn 4 (asst): 4 words
    Turn 5 (user): 1 word
    Turn 6 (asst): 5 words
    Turn 7 (asst): 4 words
    Turn 8 (asst): 3 words
    """
    turns = [
        {"speaker": "user", "message": "one two three four five six seven eight nine ten"},
        {"speaker": "assistant", "message": "how can i help you"},
        {"speaker": "user", "message": "one two three four five"},
        {"speaker": "assistant", "message": "please tell me more"},
        {"speaker": "user", "message": "tired"},
        {"speaker": "assistant", "message": "are you still there friend"},
        {"speaker": "assistant", "message": "i am waiting here"},
        {"speaker": "assistant", "message": "goodbye for now"},
    ]
    feats = extract_behavioral_features(turns)

    assert feats["total_turn_count"] == 8
    assert feats["user_turn_count"] == 3
    assert feats["assistant_turn_count"] == 5
    assert feats["last_user_turn"] == 5
    assert feats["last_speaker"] == "assistant"
    assert feats["ended_by_user"] == 0

    # Dropout gap = 8 - 5 = 3
    assert feats["dropout_gap"] == 3
    assert feats["dropout_flag"] == 1  # gap > 2

    # Negative slope because user word counts are [10, 5, 1]
    assert feats["msg_length_slope"] < 0.0


def test_fusion_vector_order_and_shape():
    turns = [
        {"speaker": "user", "message": "hello world"},
        {"speaker": "assistant", "message": "welcome"},
    ]
    feats = extract_behavioral_features(turns)
    vec = get_fusion_feature_vector(feats)

    assert vec.shape == (8,)
    assert vec.dtype == np.float32

    # Check each element corresponds to FUSION_BEHAVIORAL_COLS
    for idx, col in enumerate(FUSION_BEHAVIORAL_COLS):
        assert vec[idx] == float(feats[col])


def test_process_conversation_dataframe():
    df = pd.DataFrame({
        "conversation_id": [1, 1, 2, 2],
        "turn_number": [1, 2, 1, 2],
        "speaker": ["user", "assistant", "user", "user"],
        "message": ["I feel bad", "I am listening", "Hey", "Still here"],
    })
    summary_df, records = process_conversation_dataframe(df)
    assert len(summary_df) == 2
    assert "total_turn_count" in summary_df.columns
    assert "engagement_ratio" in summary_df.columns
    assert summary_df.loc[summary_df["conversation_id"] == 1, "total_turn_count"].values[0] == 2
