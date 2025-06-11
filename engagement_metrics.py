# engagement_metrics.py - Track session length and interactions

import datetime

# Variables to track session state
session_start = None
message_count = 0

def start_session():
    global session_start, message_count
    session_start = datetime.datetime.now()
    message_count = 0

def record_user_message():
    global message_count
    message_count += 1

def end_session():
    session_end = datetime.datetime.now()
    session_length = (session_end - session_start).total_seconds()
    print(f"Session ended. Length: {session_length:.1f}s, Messages: {message_count}")
    # TODO: log these metrics (e.g. append to a CSV or database)

# Example usage in a Gradio app:
#
# When a new user connects or conversation starts:
#   start_session()
# For each user message:
#   record_user_message()
# When conversation ends (or after each turn):
#   end_session()
