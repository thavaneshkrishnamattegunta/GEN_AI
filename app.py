from flask import (
    Flask,
    request,
    jsonify,
    render_template_string,
    redirect,
    url_for,
    session,
)
import webbrowser
import threading
import time
import os
import sqlite3
from textblob import TextBlob
import pandas as pd


app = Flask(__name__)
app.secret_key = "replace-this-with-a-random-secret"


# ---------- Simple SQLite-backed auth helpers (for assignment) ----------

DB_PATH = os.path.join(os.path.dirname(__file__), "auth.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    """Create users table if it does not exist."""
    with _get_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
            """
        )
        conn.commit()


def create_user(username: str, password: str) -> bool:
    """
    Insert a new user into the database.
    Returns True on success, False if username already exists.

    NOTE: For a classroom project we keep the password as plain text.
    In real applications you MUST hash passwords with a library such as
    werkzeug.security or passlib.
    """
    username = (username or "").strip()
    if not username or not password:
        return False

    try:
        with _get_db() as conn:
            conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password),
            )
            conn.commit()
        return True
    except sqlite3.IntegrityError:
        # UNIQUE(username) constraint failed -> user already exists
        return False


def validate_user(username: str, password: str) -> bool:
    """Return True if username/password match a row in the users table."""
    username = (username or "").strip()
    if not username or not password:
        return False

    with _get_db() as conn:
        row = conn.execute(
            "SELECT password FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        return False
    return row["password"] == password


# Create the database table on startup (safe to call multiple times)
init_auth_db()


# ---------- Core Sentiment Helpers (same logic as Streamlit version) ----------

def is_smartwatch_related(text: str) -> bool:
    """
    Checks if the text is related to smartwatch or product review.
    Returns True if relevant, False otherwise.
    """
    text_lower = text.lower()

    smartwatch_keywords = [
        "smartwatch",
        "smart watch",
        "watch",
        "wristwatch",
        "fitness tracker",
        "wearable",
        "device",
        "product",
        "purchase",
        "bought",
        "buying",
        "review",
        "reviews",
        "rating",
        "rated",
        "customer",
        "quality",
        "battery",
        "display",
        "screen",
        "band",
        "strap",
        "features",
        "app",
        "notification",
        "heart rate",
        "step",
        "activity",
        "sleep",
        "waterproof",
        "durable",
        "comfortable",
        "design",
        "price",
        "cost",
        "delivery",
        "shipping",
        "amazon",
        "recommend",
        "satisfied",
        "disappointed",
    ]

    keyword_count = sum(1 for keyword in smartwatch_keywords if keyword in text_lower)

    review_patterns = [
        "star",
        "stars",
        "out of",
        "rating",
        "would recommend",
        "great product",
        "good product",
        "bad product",
        "poor quality",
        "excellent",
        "terrible",
        "love it",
        "hate it",
        "works well",
        "doesn't work",
        "worth",
        "money",
        "value",
    ]

    pattern_count = sum(1 for pattern in review_patterns if pattern in text_lower)

    return keyword_count >= 1 or pattern_count >= 2


def classify_polarity(polarity: float) -> str:
    if polarity > 0.05:
        return "Positive"
    if polarity < -0.05:
        return "Negative"
    return "Neutral"


def apply_domain_rules(text: str, sentiment: str, polarity: float):
    """
    Applies smartwatch-specific overrides so phrases like
    'battery health is draining fast' are treated as negative.
    """
    text_lower = text.lower()

    negative_triggers = [
        "draining fast",
        "battery drain",
        "battery draining",
        "battery health is draining",
        "battery issue",
        "battery problem",
        "battery dies",
        "battery life is poor",
        "overheating",
        "laggy",
        "watch stopped working",
        "screen cracked",
        "strap broke",
    ]

    positive_triggers = [
        "battery lasts all day",
        "excellent battery",
        "great battery life",
        "long battery",
        "love the battery",
        "fast charging",
        "works flawlessly",
        "very responsive",
    ]

    if any(trigger in text_lower for trigger in negative_triggers):
        return "Negative", -abs(polarity) if polarity != 0 else -0.4

    if any(trigger in text_lower for trigger in positive_triggers):
        return "Positive", abs(polarity) if polarity != 0 else 0.4

    return sentiment, polarity


def get_sentiment(text: str):
    """
    Analyzes the sentiment of the input text using TextBlob.
    Returns (sentiment_label, polarity).
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    sentiment = classify_polarity(polarity)
    sentiment, polarity = apply_domain_rules(text, sentiment, polarity)

    return sentiment, polarity


def get_confidence(polarity: float) -> float:
    """
    Converts polarity (-1 to 1) into a 0–100 confidence value.
    """
    return round(abs(polarity) * 100, 1)


def analyze_aspects(review_text: str):
    """
    Detects smartwatch aspects mentioned in the text and scores them individually.
    Returns a list of dictionaries with aspect details.
    """
    aspect_keywords = {
        "Battery": ["battery", "charge", "charging", "power", "life"],
        "Display": ["display", "screen", "brightness", "touch", "resolution"],
        "Comfort": ["strap", "band", "comfort", "fit", "wear"],
        "Fitness Tracking": ["fitness", "heart rate", "steps", "tracking", "sleep"],
        "Notifications": ["notification", "alerts", "calls", "messages"],
        "Design & Build": ["design", "build", "quality", "durable", "style"],
        "Price & Value": ["price", "cost", "value", "worth"],
    }

    blob = TextBlob(review_text)
    sentences = blob.sentences if blob.sentences else [blob]

    aspect_results = []
    for aspect, keywords in aspect_keywords.items():
        matched_sentences = []
        for sentence in sentences:
            sentence_text = str(sentence)
            sentence_lower = sentence_text.lower()
            if any(keyword in sentence_lower for keyword in keywords):
                matched_sentences.append(sentence_text)

        if matched_sentences:
            combined = " ".join(matched_sentences)
            aspect_sentiment, aspect_polarity = get_sentiment(combined)
            aspect_results.append(
                {
                    "aspect": aspect,
                    "sentiment": aspect_sentiment,
                    "polarity": round(aspect_polarity, 3),
                    "confidence": get_confidence(aspect_polarity),
                    "evidence": matched_sentences[:2],
                }
            )

    return aspect_results


SINGLE_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>ABC X1 Smartwatch - Single Review</title>
    <style>
      :root {
        --bg-gradient-start: #dcfce7;
        --bg-gradient-end: #dbeafe;
        --card-bg: #ffffff;
        --card-border: rgba(148, 163, 184, 0.15);
        --primary: #8b5cf6;
        --primary-soft: rgba(139, 92, 246, 0.15);
        --primary-hover: #a78bfa;
        --text-main: #14532d;
        --text-muted: #166534;
        --text-dark: #052e16;
        --danger: #dc2626;
        --warning: #d97706;
        --success-green: #16a34a;
        --light-green: #86efac;
        --radius-lg: 18px;
        --radius-md: 12px;
        --shadow-soft: 0 8px 24px rgba(0, 0, 0, 0.1);
        --shadow-chip: 0 2px 8px rgba(0, 0, 0, 0.08);
        --chip-bg: rgba(255, 255, 255, 0.98);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", sans-serif;
        color: var(--text-main);
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, #cffafe 50%, var(--bg-gradient-end) 100%);
        padding: 32px 16px 40px;
      }

      .page-shell {
        max-width: 1080px;
        margin: 0 auto;
      }

      .glow-orbit {
        position: fixed;
        inset: 0;
        pointer-events: none;
        opacity: 0.4;
        background:
          radial-gradient(circle at 0% 0%, rgba(34, 197, 94, 0.2) 0, transparent 60%),
          radial-gradient(circle at 100% 20%, rgba(59, 130, 246, 0.15) 0, transparent 55%),
          radial-gradient(circle at 50% 100%, rgba(34, 197, 94, 0.12) 0, transparent 50%);
        z-index: -1;
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
      }

      .header-main {
        display: flex;
        align-items: center;
        gap: 14px;
      }

      .logo-pill {
        width: 46px;
        height: 46px;
        border-radius: 999px;
        background:
          radial-gradient(circle at 30% 0%, rgba(255, 255, 255, 0.9) 0, transparent 45%),
          conic-gradient(from 160deg, #22c55e, #8b5cf6, #3b82f6, #22c55e);
        padding: 2px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .logo-inner {
        width: 100%;
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(135deg, #dcfce7 0%, #dbeafe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
      }

      .app-title {
        font-size: 22px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: var(--text-dark);
      }

      .app-subtitle {
        font-size: 13px;
        color: var(--text-muted);
        margin-top: 4px;
      }

      .header-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        justify-content: flex-end;
      }

      .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        background: var(--chip-bg);
        color: var(--text-dark);
        font-size: 11px;
        font-weight: 500;
        border: 1px solid rgba(148, 163, 184, 0.2);
        backdrop-filter: blur(18px);
        box-shadow: var(--shadow-chip);
      }

      .chip-live {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        border-color: rgba(34, 197, 94, 0.3);
        color: #166534;
      }

      .chip-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #22c55e;
        box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.25), 0 0 8px rgba(34, 197, 94, 0.4);
        animation: pulse 2s infinite;
      }

      @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
      }

      .layout {
        display: grid;
        grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr);
        gap: 20px;
      }

      @media (max-width: 900px) {
        .layout {
          grid-template-columns: minmax(0, 1fr);
        }
        .header {
          flex-direction: column;
          align-items: flex-start;
        }
        .header-meta {
          justify-content: flex-start;
        }
      }

      .card {
        background: var(--card-bg);
        border-radius: var(--radius-lg);
        border: 1px solid var(--card-border);
        box-shadow: var(--shadow-soft);
        padding: 20px 20px 18px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(10px);
      }

      .card::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
          radial-gradient(circle at top right, rgba(139, 92, 246, 0.1) 0, transparent 60%),
          radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.08) 0, transparent 50%);
        opacity: 0.6;
      }

      .card-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 12px;
        color: var(--text-dark);
        letter-spacing: -0.01em;
      }

      .label {
        font-weight: 500;
        font-size: 13px;
        color: var(--text-dark);
      }

      textarea {
        width: 100%;
        margin-top: 6px;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1.5px solid rgba(148, 163, 184, 0.3);
        background: #fafafa;
        color: var(--text-dark);
        resize: vertical;
        min-height: 140px;
        font-size: 13px;
        line-height: 1.5;
        transition: all 0.2s ease;
      }

      textarea:focus-visible {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15), 0 2px 8px rgba(139, 92, 246, 0.1);
        background: #ffffff;
      }

      .form-footer {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        margin-top: 10px;
        flex-wrap: wrap;
      }

      .checkbox-row {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 12px;
        color: var(--text-dark);
      }

      .checkbox-row input[type="checkbox"] {
        accent-color: var(--primary);
      }

      .button-row {
        display: flex;
        align-items: center;
        gap: 8px;
        flex-wrap: wrap;
      }

      .btn-primary {
        border: none;
        cursor: pointer;
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: #ffffff;
        background: radial-gradient(circle at 0% 0%, rgba(167, 139, 250, 0.8) 0, transparent 60%),
                    linear-gradient(135deg, var(--primary) 0%, #a78bfa 50%, var(--primary-hover) 100%);
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4), 0 2px 4px rgba(139, 92, 246, 0.2);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
      }

      .btn-primary:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5), 0 4px 8px rgba(139, 92, 246, 0.3);
      }

      .btn-primary:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4);
      }

      .btn-ghost {
        font-size: 12px;
        border-radius: 999px;
        padding: 7px 12px;
        border: 1.5px solid rgba(148, 163, 184, 0.3);
        color: var(--text-dark);
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(16px);
        font-weight: 500;
        transition: all 0.2s ease;
      }

      .btn-ghost span {
        opacity: 0.95;
      }

      .btn-ghost:hover {
        color: var(--text-dark);
        border-color: rgba(139, 92, 246, 0.5);
        background: rgba(255, 255, 255, 1);
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.15);
        transform: translateY(-1px);
      }

      .alert {
        margin-top: 10px;
        padding: 8px 10px;
        border-radius: var(--radius-md);
        font-size: 12px;
        display: flex;
        align-items: flex-start;
        gap: 8px;
      }

      .alert-icon {
        margin-top: 1px;
      }

      .alert-warning {
        background: linear-gradient(135deg, rgba(250, 204, 21, 0.15), rgba(251, 191, 36, 0.08));
        border: 1px solid rgba(217, 119, 6, 0.4);
        color: #b45309;
      }

      .alert-error {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(220, 38, 38, 0.4);
        color: #991b1b;
      }

      .alert p {
        margin: 0;
      }

      .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 11px;
        line-height: 1;
        border: 1px solid transparent;
      }

      .badge-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: currentColor;
      }

      .badge-pos {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(22, 163, 74, 0.15));
        color: #166534;
        border-color: rgba(34, 197, 94, 0.5);
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(34, 197, 94, 0.2);
      }

      .badge-neg {
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.2), rgba(185, 28, 28, 0.15));
        color: #991b1b;
        border-color: rgba(220, 38, 38, 0.5);
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.2);
      }

      .badge-neu {
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.2), rgba(100, 116, 139, 0.15));
        color: #475569;
        border-color: rgba(148, 163, 184, 0.5);
        font-weight: 700;
        box-shadow: 0 2px 6px rgba(148, 163, 184, 0.15);
      }

      .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0 16px;
      }

      .metric {
        padding: 10px 12px;
        border-radius: var(--radius-md);
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
        border: 1.5px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
      }

      .metric:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
      }

      .metric-label {
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }

      .metric-value {
        font-size: 16px;
        margin-top: 4px;
        color: var(--text-dark);
        font-weight: 700;
      }

      .sentiment-positive {
        color: #166534 !important;
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        padding: 5px 10px;
        border-radius: 8px;
        display: inline-block;
        border: 1.5px solid rgba(34, 197, 94, 0.4);
        box-shadow: 0 2px 6px rgba(34, 197, 94, 0.2);
        font-weight: 700;
      }

      .sentiment-negative {
        color: #991b1b !important;
        background: linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(185, 28, 28, 0.1));
        padding: 5px 10px;
        border-radius: 8px;
        display: inline-block;
        border: 1.5px solid rgba(220, 38, 38, 0.4);
        box-shadow: 0 2px 6px rgba(220, 38, 38, 0.2);
        font-weight: 700;
      }

      .sentiment-neutral {
        color: #475569 !important;
        background: linear-gradient(135deg, rgba(148, 163, 184, 0.15), rgba(100, 116, 139, 0.1));
        padding: 5px 10px;
        border-radius: 8px;
        display: inline-block;
        border: 1.5px solid rgba(148, 163, 184, 0.4);
        box-shadow: 0 2px 6px rgba(148, 163, 184, 0.15);
        font-weight: 700;
      }

      .polarity-value {
        font-family: 'Courier New', monospace;
        font-weight: 700;
      }

      .bar-section-title {
        font-size: 13px;
        margin-bottom: 6px;
        color: var(--text-dark);
        font-weight: 600;
      }

      .bar-caption {
        font-size: 11px;
        color: var(--text-muted);
        margin-bottom: 8px;
        font-weight: 500;
      }

      .bar-container {
        height: 16px;
        background: linear-gradient(90deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 999px;
        overflow: hidden;
        border: 1.5px solid rgba(148, 163, 184, 0.25);
        position: relative;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
      }

      .bar-fill {
        height: 100%;
        transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
      }

      .bar-fill::after {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
      }

      .bar-pos {
        background: linear-gradient(90deg, #22c55e 0%, #4ade80 50%, #86efac 100%);
        box-shadow: 0 0 12px rgba(34, 197, 94, 0.4), inset 0 1px 2px rgba(255, 255, 255, 0.3);
      }

      .bar-neg {
        background: linear-gradient(90deg, #ef4444 0%, #f87171 50%, #fca5a5 100%);
        box-shadow: 0 0 12px rgba(220, 38, 38, 0.4), inset 0 1px 2px rgba(255, 255, 255, 0.3);
      }

      .bar-neu {
        background: linear-gradient(90deg, #64748b 0%, #94a3b8 50%, #cbd5e1 100%);
        box-shadow: 0 0 12px rgba(100, 116, 139, 0.3), inset 0 1px 2px rgba(255, 255, 255, 0.3);
      }

      .aspects-list {
        list-style: none;
        padding: 0;
        margin: 8px 0 0;
        display: flex;
        flex-direction: column;
        gap: 8px;
        font-size: 13px;
      }

      .aspect-row {
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: flex-start;
      }

      .aspect-main {
        display: flex;
        flex-direction: column;
        gap: 2px;
      }

      .aspect-label {
        font-weight: 600;
        color: var(--text-dark);
      }

      .aspect-meta {
        font-size: 11px;
        color: var(--text-muted);
        font-weight: 500;
      }

      .aspect-evidence {
        font-size: 11px;
        color: var(--text-muted);
        margin-top: 2px;
      }

      .aspect-pill {
        white-space: nowrap;
        padding: 4px 8px;
        border-radius: 999px;
        background: #f9fafb;
        border: 1px solid rgba(148, 163, 184, 0.3);
        font-size: 11px;
        color: var(--text-dark);
        font-weight: 500;
      }

      .empty-hint {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 4px;
      }
    </style>
  </head>
  <body>
    <div class="glow-orbit"></div>
    <div class="page-shell">
      <header class="header">
        <div class="header-main">
          <div class="logo-pill">
            <div class="logo-inner">⌚</div>
          </div>
          <div>
            <div class="app-title">ABC X1 Smartwatch – Sentiment Studio</div>
            <div class="app-subtitle">Analyze customer reviews with domain‑tuned sentiment and aspect insights.</div>
          </div>
        </div>
        <div class="header-meta">
          <div class="chip chip-live">
            <span class="chip-dot"></span>
            <span>Live analyzer</span>
          </div>
          <div class="chip">
            <span>Smartwatch domain model</span>
          </div>
          {% if current_user %}
            <div class="chip">
              <span>Logged in as <b>{{ current_user }}</b></span>
            </div>
            <form method="post" action="{{ url_for('logout') }}" style="margin:0;">
              <button type="submit" class="btn-ghost" style="background:#166534; color:#ffffff; border-color:#166534;">
                <span>Logout</span>
              </button>
            </form>
          {% endif %}
        </div>
      </header>

      <div class="layout">
        <div class="card">
          <div class="card-title">Single review</div>
          <form method="post">
            <div>
              <label class="label" for="review_text">Review text</label><br>
              <textarea id="review_text" name="review_text" rows="6"
                placeholder="Type your smartwatch review here...">{{ review_text }}</textarea>
            </div>

            {% if enforce_warning %}
              <div class="alert alert-warning">
                <div class="alert-icon">⚠</div>
                <p>
                  The text doesn't look like a smartwatch / product review.
                  This workspace is tuned specifically for <b>ABC X1 smartwatch feedback</b>.
                  Check that your text describes the device, its features or your experience using it.
                </p>
              </div>
            {% endif %}

            {% if error %}
              <div class="alert alert-error">
                <div class="alert-icon">⛔</div>
                <p>{{ error }}</p>
              </div>
            {% endif %}

            <div class="form-footer">
              <label class="checkbox-row">
                <input type="checkbox" name="proceed_anyway" value="1"
                  {% if proceed_anyway %}checked{% endif %}>
                <span>Proceed even if not clearly smartwatch‑related</span>
              </label>
              <div class="button-row">
                <button class="btn-primary" type="submit">
                  <span>🔍</span>
                  <span>Analyze review</span>
                </button>
                <a href="/batch" class="btn-ghost">
                  <span>Batch analysis</span>
                  <span>▶</span>
                </a>
              </div>
            </div>
          </form>
        </div>

        <div class="card">
          <div class="card-title">Analysis results</div>
          {% if sentiment %}
            <div style="display:flex; align-items:center; justify-content:space-between; gap:8px; flex-wrap:wrap;">
              <div>
                {% if relevant %}
                  <span class="badge badge-pos">
                    <span class="badge-dot"></span>
                    <span>Smartwatch‑related</span>
                  </span>
                {% elif relevant is not none %}
                  <span class="badge badge-neu">
                    <span class="badge-dot"></span>
                    <span>Non‑smartwatch text (testing mode)</span>
                  </span>
                {% endif %}
              </div>
            </div>

            <div class="metric-grid">
              <div class="metric">
                <div class="metric-label">Overall sentiment</div>
                <div class="metric-value sentiment-{{ sentiment.lower() }}">{{ sentiment }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">Polarity</div>
                <div class="metric-value polarity-value">{{ polarity }}</div>
              </div>
              <div class="metric">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{{ confidence }}%</div>
              </div>
            </div>

            <div style="margin-top:8px; margin-bottom:14px;">
              <div class="bar-section-title">Polarity visualization</div>
              <p class="bar-caption">
                <span style="color:#991b1b; font-weight:600;">-1.0 (Negative)</span> &larr; 
                <span style="color:#475569; font-weight:600;">0.0 (Neutral)</span> &rarr; 
                <span style="color:#166534; font-weight:600;">1.0 (Positive)</span>
              </p>
              <div class="bar-container">
                {% set pct = (polarity + 1) / 2 * 100 %}
                {% if sentiment == "Positive" %}
                  {% set bar_class = "bar-pos" %}
                {% elif sentiment == "Negative" %}
                  {% set bar_class = "bar-neg" %}
                {% else %}
                  {% set bar_class = "bar-neu" %}
                {% endif %}
                <div class="bar-fill {{ bar_class }}" style="width: {{ pct }}%;"></div>
              </div>
              <div style="display:flex; justify-content:space-between; margin-top:4px; font-size:10px; color:var(--text-muted);">
                <span>-1.0</span>
                <span style="font-weight:600; color:var(--text-dark);">Current: {{ polarity }}</span>
                <span>1.0</span>
              </div>
            </div>

            <div>
              <div class="bar-section-title">Aspect insights</div>
              {% if aspects and aspects|length > 0 %}
                <ul class="aspects-list">
                  {% for a in aspects %}
                    <li class="aspect-row">
                      <div class="aspect-main">
                        <span class="aspect-label">{{ a.aspect }}</span>
                        <span class="aspect-meta">
                          {{ a.sentiment }} &nbsp;•&nbsp;
                          polarity {{ a.polarity }}, {{ a.confidence }}%
                        </span>
                        {% if a.evidence %}
                          <div class="aspect-evidence">
                            {{ a.evidence | join("; ") }}
                          </div>
                        {% endif %}
                      </div>
                      <div class="aspect-pill">
                        Aspect
                      </div>
                    </li>
                  {% endfor %}
                </ul>
              {% else %}
                <p class="empty-hint">
                  No explicit aspects detected. Mention battery, display, comfort, notifications or price
                  to surface more granular insights.
                </p>
              {% endif %}
            </div>
          {% else %}
            <p class="empty-hint">
              Results will appear here once you analyze a review.
            </p>
          {% endif %}
        </div>
      </div>
    </div>
  </body>
</html>
"""


BATCH_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>ABC X1 Smartwatch - Batch Analysis</title>
    <style>
      body {
        margin: 0;
        min-height: 100vh;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", sans-serif;
        background: linear-gradient(135deg, #dcfce7 0%, #cffafe 50%, #dbeafe 100%);
        padding: 32px 16px 40px;
        color: #14532d;
      }

      .page-shell {
        max-width: 1100px;
        margin: 0 auto;
      }

      .header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 20px;
      }

      .title-block {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .title-main {
        font-size: 21px;
        font-weight: 600;
        letter-spacing: 0.02em;
        color: #14532d;
      }

      .title-sub {
        font-size: 13px;
        color: #15803d;
      }

      .btn-ghost {
        font-size: 12px;
        border-radius: 999px;
        padding: 7px 11px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        color: #14532d;
        text-decoration: none;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(16px);
        transition: border-color 120ms ease, color 120ms ease, background 120ms ease;
      }

      .btn-ghost:hover {
        color: #14532d;
        border-color: rgba(124, 58, 237, 0.6);
        background: rgba(255, 255, 255, 1);
      }

      .card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        padding: 20px 20px 16px;
        margin-bottom: 16px;
        backdrop-filter: blur(10px);
      }

      .card-title {
        font-size: 16px;
        font-weight: 700;
        margin-bottom: 10px;
        color: #052e16;
        letter-spacing: -0.01em;
      }

      label {
        font-size: 13px;
        color: #14532d;
        font-weight: 600;
      }

      input[type="text"] {
        margin-top: 4px;
        padding: 7px 9px;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: #f9fafb;
        color: #14532d;
        width: 260px;
        font-size: 13px;
      }

      input[type="text"]:focus-visible {
        outline: none;
        border-color: #7c3aed;
        box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.2);
        background: #ffffff;
      }

      input[type="file"] {
        margin-top: 4px;
        font-size: 13px;
        color: #14532d;
      }

      .btn-primary {
        border: none;
        cursor: pointer;
        padding: 10px 18px;
        border-radius: 999px;
        font-size: 13px;
        font-weight: 600;
        color: #ffffff;
        background: radial-gradient(circle at 0% 0%, rgba(167, 139, 250, 0.8) 0, transparent 60%),
                    linear-gradient(135deg, #8b5cf6 0%, #a78bfa 50%, #c4b5fd 100%);
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4), 0 2px 4px rgba(139, 92, 246, 0.2);
        display: inline-flex;
        align-items: center;
        gap: 6px;
        transition: all 0.2s ease;
        margin-top: 8px;
      }

      .btn-primary:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5), 0 4px 8px rgba(139, 92, 246, 0.3);
      }

      .btn-primary:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4);
      }

      .alert-error {
        margin-top: 10px;
        padding: 8px 10px;
        border-radius: 12px;
        font-size: 12px;
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(220, 38, 38, 0.4);
        color: #991b1b;
      }

      .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-top: 6px;
      }

      .summary-item {
        padding: 10px 12px;
        border-radius: 12px;
        background: linear-gradient(135deg, #fafafa 0%, #f5f5f5 100%);
        border: 1.5px solid rgba(148, 163, 184, 0.2);
        font-size: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
      }

      .summary-item:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.06);
      }

      .summary-label {
        color: #15803d;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 11px;
      }

      .summary-value {
        margin-top: 4px;
        font-size: 14px;
        color: #052e16;
        font-weight: 700;
      }

      table {
        border-collapse: collapse;
        width: 100%;
        font-size: 12px;
      }

      th, td {
        border: 1px solid rgba(31, 41, 55, 0.95);
        padding: 6px 8px;
        vertical-align: top;
      }

      th {
        background: linear-gradient(135deg, #f9fafb, #f1f5f9);
        font-weight: 600;
        color: #14532d;
        position: sticky;
        top: 0;
        z-index: 1;
      }

      td {
        color: #14532d;
      }

      tbody tr:nth-child(even) {
        background: #f9fafb;
      }

      tbody tr:nth-child(odd) {
        background: #ffffff;
      }

      .table-scroll {
        max-height: 380px;
        overflow-y: auto;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.3);
      }
    </style>
  </head>
  <body>
    <div class="page-shell">
      <header class="header">
        <div class="title-block">
          <div class="title-main">⌚ ABC X1 Smartwatch – Batch Sentiment</div>
          <div class="title-sub">Upload a CSV of reviews to analyze sentiment and relevance at scale.</div>
        </div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <a href="/" class="btn-ghost">
            <span>◀</span>
            <span>Back to single review</span>
          </a>
          {% if current_user %}
            <span style="font-size:12px; color:#14532d; font-weight:500;">Logged in as <b>{{ current_user }}</b></span>
            <form method="post" action="{{ url_for('logout') }}" style="margin:0;">
              <button type="submit" class="btn-ghost" style="background:#166534; color:#ffffff; border-color:#166534;">
                <span>Logout</span>
              </button>
            </form>
          {% endif %}
        </div>
      </header>

      <div class="card">
        <div class="card-title">Upload reviews</div>
        <form method="post" enctype="multipart/form-data">
          <div>
            <label for="file"><b>CSV file</b></label><br>
            <input type="file" id="file" name="file" accept=".csv">
          </div>
          <div style="margin-top: 8px;">
            <label for="text_column"><b>Text column (optional)</b></label><br>
            <input type="text" id="text_column" name="text_column"
              placeholder="e.g. reviews.text, text, review">
          </div>

          {% if error %}
            <div class="alert-error">{{ error }}</div>
          {% endif %}

          <button type="submit" class="btn-primary">
            <span>🔍</span>
            <span>Analyze all reviews</span>
          </button>
        </form>
      </div>

      {% if summary %}
        <div class="card">
          <div class="card-title">Batch analysis results</div>
          <div class="summary-grid">
            <div class="summary-item">
              <div class="summary-label">Total rows</div>
              <div class="summary-value">{{ summary.total_rows }}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Text column</div>
              <div class="summary-value">{{ summary.text_column }}</div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Relevant / Irrelevant</div>
              <div class="summary-value">
                {{ summary.relevant_count }} relevant • {{ summary.irrelevant_count }} filtered out
              </div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Sentiment (relevant only)</div>
              <div class="summary-value">
                <span style="color:#166534; font-weight:700;">Pos: {{ summary.sentiment_distribution.Positive }}</span>,
                <span style="color:#475569; font-weight:700;">Neu: {{ summary.sentiment_distribution.Neutral }}</span>,
                <span style="color:#991b1b; font-weight:700;">Neg: {{ summary.sentiment_distribution.Negative }}</span>
              </div>
            </div>
            <div class="summary-item">
              <div class="summary-label">Avg polarity (relevant only)</div>
              <div class="summary-value">
                {% if summary.average_polarity is not none %}
                  {{ summary.average_polarity }}
                {% else %}
                  N/A
                {% endif %}
              </div>
            </div>
          </div>
        </div>

        {% if rows %}
          <div class="card">
            <div class="card-title">Sample of analyzed rows (first 200)</div>
            <div class="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Review</th>
                    <th>Sentiment</th>
                    <th>Polarity</th>
                    <th>Confidence</th>
                    <th>Relevant</th>
                  </tr>
                </thead>
                <tbody>
                  {% for r in rows %}
                    <tr>
                      <td>{{ r.review }}</td>
                      <td>
                        {% if r.sentiment == "Positive" %}
                          <span style="color:#166534; font-weight:600; background:rgba(34, 197, 94, 0.1); padding:2px 6px; border-radius:4px; border:1px solid rgba(34, 197, 94, 0.3);">{{ r.sentiment }}</span>
                        {% elif r.sentiment == "Negative" %}
                          <span style="color:#991b1b; font-weight:600; background:rgba(220, 38, 38, 0.1); padding:2px 6px; border-radius:4px; border:1px solid rgba(220, 38, 38, 0.3);">{{ r.sentiment }}</span>
                        {% elif r.sentiment == "Neutral" %}
                          <span style="color:#475569; font-weight:600; background:rgba(148, 163, 184, 0.1); padding:2px 6px; border-radius:4px; border:1px solid rgba(148, 163, 184, 0.3);">{{ r.sentiment }}</span>
                        {% else %}
                          {{ r.sentiment }}
                        {% endif %}
                      </td>
                      <td style="font-family:'Courier New', monospace; font-weight:600;">
                        {% if r.polarity is not none %}{{ r.polarity }}{% else %}-{% endif %}
                      </td>
                      <td>{% if r.confidence is not none %}{{ r.confidence }}%{% else %}-{% endif %}</td>
                      <td>
                        {% if r.relevant == "Yes" %}
                          <span style="color:#166534; font-weight:600;">{{ r.relevant }}</span>
                        {% elif r.relevant == "No" %}
                          <span style="color:#991b1b; font-weight:600;">{{ r.relevant }}</span>
                        {% else %}
                          {{ r.relevant }}
                        {% endif %}
                      </td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
        {% endif %}
      {% endif %}
    </div>
  </body>
</html>
"""


AUTH_TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>{{ page_title }}</title>
    <style>
      :root {
        --bg-gradient-start: #dcfce7;
        --bg-gradient-end: #dbeafe;
        --primary: #8b5cf6;
        --primary-hover: #a78bfa;
        --text-main: #14532d;
        --text-muted: #15803d;
        --text-dark: #052e16;
        --card-border: rgba(148, 163, 184, 0.15);
      }

      * {
        box-sizing: border-box;
      }

      body {
        margin: 0;
        min-height: 100vh;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                     "Segoe UI", sans-serif;
        color: var(--text-main);
        background: linear-gradient(135deg, var(--bg-gradient-start) 0%, #cffafe 50%, var(--bg-gradient-end) 100%);
        padding: 24px 16px;
      }

      .shell {
        width: 100%;
        max-width: 440px;
      }

      .card {
        background: #ffffff;
        border-radius: 18px;
        border: 1px solid var(--card-border);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
        padding: 24px 24px 20px;
        backdrop-filter: blur(10px);
      }

      .title {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 6px;
        color: var(--text-dark);
        letter-spacing: -0.01em;
      }

      .subtitle {
        font-size: 13px;
        color: var(--text-muted);
        margin-bottom: 16px;
      }

      label {
        display: block;
        font-size: 13px;
        color: var(--text-dark);
        margin-bottom: 4px;
        font-weight: 500;
      }

      input[type="text"],
      input[type="password"] {
        width: 100%;
        padding: 10px 12px;
        border-radius: 12px;
        border: 1.5px solid rgba(148, 163, 184, 0.3);
        background: #fafafa;
        color: var(--text-dark);
        font-size: 13px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
      }

      input[type="text"]:focus-visible,
      input[type="password"]:focus-visible {
        outline: none;
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.15), 0 2px 8px rgba(139, 92, 246, 0.1);
        background: #ffffff;
      }

      .btn-primary {
        width: 100%;
        border: none;
        cursor: pointer;
        padding: 11px 18px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        background: radial-gradient(circle at 0% 0%, rgba(167, 139, 250, 0.8) 0, transparent 60%),
                    linear-gradient(135deg, var(--primary) 0%, #a78bfa 50%, var(--primary-hover) 100%);
        box-shadow: 0 4px 14px rgba(139, 92, 246, 0.4), 0 2px 4px rgba(139, 92, 246, 0.2);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        transition: all 0.2s ease;
        margin-top: 6px;
      }

      .btn-primary:hover {
        transform: translateY(-2px);
        filter: brightness(1.08);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.5), 0 4px 8px rgba(139, 92, 246, 0.3);
      }

      .btn-primary:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(139, 92, 246, 0.4);
      }

      .muted {
        font-size: 12px;
        color: var(--text-muted);
        margin-top: 10px;
        text-align: center;
      }

      .muted a {
        color: #8b5cf6;
        text-decoration: none;
        font-weight: 600;
      }

      .muted a:hover {
        text-decoration: underline;
      }

      .alert {
        font-size: 12px;
        border-radius: 10px;
        padding: 7px 9px;
        margin-bottom: 8px;
      }

      .alert-error {
        background: linear-gradient(135deg, rgba(248, 113, 113, 0.15), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(220, 38, 38, 0.4);
        color: #991b1b;
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <div class="title">{{ heading }}</div>
        <div class="subtitle">{{ subheading }}</div>

        {% if error %}
          <div class="alert alert-error">{{ error }}</div>
        {% endif %}

        <form method="post">
          <label for="username">Username</label>
          <input
            id="username"
            name="username"
            type="text"
            value="{{ username or '' }}"
            autocomplete="username"
            required
          >

          <label for="password">Password</label>
          <input
            id="password"
            name="password"
            type="password"
            autocomplete="current-password"
            required
          >

          <button class="btn-primary" type="submit">
            <span>{{ button_label }}</span>
          </button>
        </form>

        <div class="muted">
          {{ footer_text | safe }}
        </div>
      </div>
    </div>
  </body>
</html>
"""


@app.route("/login", methods=["GET", "POST"])
def login():
    """
    Very basic login page with validation.
    Checks that the username exists in the database and that the
    password exactly matches the stored value.
    """
    if session.get("user"):
        return redirect(url_for("index"))

    error = None
    username = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            error = "Please enter both username and password."
        elif not validate_user(username, password):
            error = "Invalid username or password."
        else:
            session["user"] = username
            return redirect(url_for("index"))

    footer_text = f'New here? <a href="{url_for("signup")}">Create an account</a>.'

    return render_template_string(
        AUTH_TEMPLATE,
        page_title="Login",
        heading="Sign in",
        subheading="Use your account to access the smartwatch sentiment analyzer.",
        button_label="Login",
        footer_text=footer_text,
        error=error,
        username=username,
    )


@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    Very basic signup page with validation.
    First-time users must create an account here. Username must be
    unique and a non-empty password is required.
    """
    if session.get("user"):
        return redirect(url_for("index"))

    error = None
    username = ""

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            error = "Please choose a username and password."
        elif not create_user(username, password):
            error = "That username is already taken. Please pick another one."
        else:
            session["user"] = username
            return redirect(url_for("index"))

    footer_text = f'Already have an account? <a href="{url_for("login")}">Login</a>.'

    return render_template_string(
        AUTH_TEMPLATE,
        page_title="Sign up",
        heading="Create an account",
        subheading="Set up a basic account for this smartwatch project (demo only).",
        button_label="Sign up",
        footer_text=footer_text,
        error=error,
        username=username,
    )


@app.post("/logout")
def logout():
    """Clear the current session and return to the login screen."""
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Simple HTML interface for single-review analysis.
    Mirrors the Streamlit behaviour: relevance check, sentiment,
    polarity, confidence and aspect-level insights.
    """
    if "user" not in session:
        return redirect(url_for("login"))

    context = {
        "review_text": "",
        "relevant": None,
        "enforce_warning": False,
        "sentiment": None,
        "polarity": None,
        "confidence": None,
        "aspects": [],
        "proceed_anyway": False,
        "error": None,
        "current_user": session.get("user"),
    }

    if request.method == "POST":
        review_text = (request.form.get("review_text") or "").strip()
        proceed_anyway = bool(request.form.get("proceed_anyway"))
        context["review_text"] = review_text
        context["proceed_anyway"] = proceed_anyway

        if not review_text:
            context["error"] = "Please enter a review to analyze."
            return render_template_string(SINGLE_TEMPLATE, **context)

        is_relevant = is_smartwatch_related(review_text)
        context["relevant"] = is_relevant

        # If it doesn't look like a smartwatch review and user didn't opt in, stop here
        if (not is_relevant) and (not proceed_anyway):
            context["enforce_warning"] = True
            return render_template_string(SINGLE_TEMPLATE, **context)

        sentiment, polarity = get_sentiment(review_text)
        aspects = analyze_aspects(review_text)

        context["sentiment"] = sentiment
        context["polarity"] = round(polarity, 3)
        context["confidence"] = get_confidence(polarity)
        context["aspects"] = aspects

    return render_template_string(SINGLE_TEMPLATE, **context)


@app.route("/batch", methods=["GET", "POST"])
def batch():
    """
    HTML interface for batch CSV analysis.
    Reuses the same logic as the /api/analyze-batch endpoint, but
    renders a very simple page with summary + sample rows.
    """
    if "user" not in session:
        return redirect(url_for("login"))

    context = {
        "summary": None,
        "rows": None,
        "error": None,
        "current_user": session.get("user"),
    }

    if request.method == "POST":
        file = request.files.get("file")
        text_column = request.form.get("text_column") or None

        if file is None or file.filename == "":
            context["error"] = "Please upload a CSV file."
            return render_template_string(BATCH_TEMPLATE, **context)

        try:
            df = pd.read_csv(file)
        except Exception as exc:
            context["error"] = f"Could not read CSV: {exc}"
            return render_template_string(BATCH_TEMPLATE, **context)

        text_columns = [c for c in df.columns if df[c].dtype == "object"]
        if not text_columns:
            context["error"] = "No text-like columns found in the CSV."
            return render_template_string(BATCH_TEMPLATE, **context)

        if text_column:
            if text_column not in df.columns:
                context["error"] = f"text_column '{text_column}' not found."
                return render_template_string(BATCH_TEMPLATE, **context)
            text_col = text_column
        else:
            preferred = ["reviews.text", "text", "review", "reviews", "comment", "comments"]
            text_col = None
            for col in preferred:
                if col in text_columns:
                    text_col = col
                    break
            if text_col is None:
                text_col = text_columns[0]

        results = []
        relevant_count = 0
        irrelevant_count = 0

        for raw in df[text_col]:
            if pd.notna(raw) and str(raw).strip():
                review_str = str(raw)
                is_relevant = is_smartwatch_related(review_str)
                if is_relevant:
                    relevant_count += 1
                    sentiment, polarity = get_sentiment(review_str)
                    results.append(
                        {
                            "review": review_str,
                            "sentiment": sentiment,
                            "polarity": round(polarity, 3),
                            "confidence": get_confidence(polarity),
                            "relevant": "Yes",
                        }
                    )
                else:
                    irrelevant_count += 1
                    results.append(
                        {
                            "review": review_str,
                            "sentiment": "Not analyzed",
                            "polarity": None,
                            "confidence": None,
                            "relevant": "No",
                        }
                    )
            else:
                results.append(
                    {
                        "review": None,
                        "sentiment": "Neutral",
                        "polarity": 0.0,
                        "confidence": None,
                        "relevant": "N/A",
                    }
                )

        # Compute summary on relevant reviews only
        relevant_reviews = [r for r in results if r["relevant"] == "Yes"]
        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
        avg_polarity = None

        if relevant_reviews:
            polys = []
            for r in relevant_reviews:
                s = r["sentiment"]
                if s in sentiment_counts:
                    sentiment_counts[s] += 1
                if r["polarity"] is not None:
                    polys.append(r["polarity"])
            if polys:
                avg_polarity = round(sum(polys) / len(polys), 3)

        context["summary"] = {
            "total_rows": len(df),
            "text_column": text_col,
            "relevant_count": relevant_count,
            "irrelevant_count": irrelevant_count,
            "sentiment_distribution": sentiment_counts,
            "average_polarity": avg_polarity,
        }
        # Show up to first 200 rows in the HTML table for simplicity
        context["rows"] = results[:200]

    return render_template_string(BATCH_TEMPLATE, **context)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/analyze-review")
def api_analyze_review():
    """
    JSON body:
    {
      "text": "review text here",
      "enforce_smartwatch": true | false   # optional, default: true
    }
    """
    data = request.get_json(silent=True) or {}
    review_text = (data.get("text") or "").strip()
    enforce = data.get("enforce_smartwatch", True)

    if not review_text:
        return jsonify({"error": "Field 'text' is required."}), 400

    is_relevant = is_smartwatch_related(review_text)
    if enforce and not is_relevant:
        return (
            jsonify(
                {
                    "relevant": False,
                    "message": "Text does not appear to be a smartwatch/product review.",
                }
            ),
            200,
        )

    sentiment, polarity = get_sentiment(review_text)
    aspects = analyze_aspects(review_text)

    return jsonify(
        {
            "relevant": bool(is_relevant),
            "sentiment": sentiment,
            "polarity": round(polarity, 3),
            "confidence": get_confidence(polarity),
            "aspects": aspects,
        }
    )


@app.post("/api/analyze-batch")
def api_analyze_batch():
    """
    Multipart form-data:
      - file: CSV file
      - text_column: optional, column name with review text
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded under field 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Uploaded file has no name."}), 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400

    # Choose text column
    requested_col = request.form.get("text_column")
    text_columns = [c for c in df.columns if df[c].dtype == "object"]

    if not text_columns:
        return jsonify({"error": "No text-like columns found in the CSV."}), 400

    if requested_col:
        if requested_col not in df.columns:
            return jsonify({"error": f"text_column '{requested_col}' not found."}), 400
        text_col = requested_col
    else:
        # Auto-detect if not provided
        preferred = ["reviews.text", "text", "review", "reviews", "comment", "comments"]
        text_col = None
        for col in preferred:
            if col in text_columns:
                text_col = col
                break
        if text_col is None:
            text_col = text_columns[0]

    results = []
    relevant_count = 0
    irrelevant_count = 0

    for raw in df[text_col]:
        if pd.notna(raw) and str(raw).strip():
            review_str = str(raw)
            is_relevant = is_smartwatch_related(review_str)
            if is_relevant:
                relevant_count += 1
                sentiment, polarity = get_sentiment(review_str)
                results.append(
                    {
                        "review": review_str,
                        "sentiment": sentiment,
                        "polarity": round(polarity, 3),
                        "confidence": get_confidence(polarity),
                        "relevant": True,
                    }
                )
            else:
                irrelevant_count += 1
                results.append(
                    {
                        "review": review_str,
                        "sentiment": None,
                        "polarity": None,
                        "confidence": None,
                        "relevant": False,
                    }
                )
        else:
            results.append(
                {
                    "review": None,
                    "sentiment": None,
                    "polarity": None,
                    "confidence": None,
                    "relevant": None,
                }
            )

    # Compute simple stats on relevant reviews
    relevant_reviews = [r for r in results if r["relevant"]]
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    avg_polarity = None

    if relevant_reviews:
        polys = []
        for r in relevant_reviews:
            s = r["sentiment"]
            if s in sentiment_counts:
                sentiment_counts[s] += 1
            if r["polarity"] is not None:
                polys.append(r["polarity"])
        if polys:
            avg_polarity = round(sum(polys) / len(polys), 3)

    summary = {
        "total_rows": len(df),
        "text_column": text_col,
        "relevant_count": relevant_count,
        "irrelevant_count": irrelevant_count,
        "sentiment_distribution": sentiment_counts,
        "average_polarity": avg_polarity,
    }

    return jsonify({"summary": summary, "results": results})


def _open_browser():
    """Open the default web browser to the main page after a short delay."""
    time.sleep(1.0)
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    # Only auto-open the browser for the reloader main process to avoid double-open
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=_open_browser, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=True)


