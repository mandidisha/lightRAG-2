import json
import os
from datetime import datetime

def process_interaction(
    question: str,
    answer: str,
    explanation: str,
    clarity_score: int,
    helpfulness_score: int
):
    """
    Process the user's interaction and feedback.
    Appends each record to `feedback.json` in the working directory.
    """
    # Build one entry
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "question": question,
        "answer": answer,
        "explanation": explanation,
        "ratings": {
            "clarity": clarity_score,
            "helpfulness": helpfulness_score
        }
    }

    # Load existing feedback list (or start new)
    feedback_path = "feedback.json"
    if os.path.exists(feedback_path):
        try:
            with open(feedback_path, "r", encoding="utf-8") as f:
                all_feedback = json.load(f)
        except json.JSONDecodeError:
            all_feedback = []
    else:
        all_feedback = []

    # Append and write back
    all_feedback.append(entry)
    with open(feedback_path, "w", encoding="utf-8") as f:
        json.dump(all_feedback, f, indent=2, ensure_ascii=False)

    # Optionally, log to console for dev
    print(f"[Feedback saved] {entry['timestamp']} – question: {question!r}")
