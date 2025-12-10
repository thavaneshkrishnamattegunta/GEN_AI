# GEN_AI
# ⌚ Smartwatch Sentiment Analyzer

A Gen-AI–inspired NLP project that analyzes **smartwatch product reviews** and classifies them as **Positive, Negative, or Neutral**.  
The system also checks whether the text is actually about a smartwatch and extracts **aspect-based insights** such as Battery, Display, Comfort, Notifications, Design & Build, and Price. :contentReference[oaicite:0]{index=0}

---

## 🎯 Problem Statement

Smartwatch brands receive thousands of reviews on e-commerce platforms. These reviews:
- Contain mixed or unclear feedback
- Talk about many aspects (battery, display, strap, etc.)
- Are hard to manually analyse at scale

**Goal:**  
Build a tool that **automatically understands customer sentiment** from smartwatch reviews so product teams can:
- Track customer satisfaction
- Find common complaints
- Improve product design and features 

---

## ✅ Objectives

- Develop a **user-friendly sentiment analysis tool**
- Automatically classify reviews into **Positive / Neutral / Negative**
- Detect whether a review is **really about a smartwatch**
- Provide **aspect-wise sentiment** (Battery, Display, Fitness Tracking, etc.)
- Support:
  - **Single review analysis**
  - **Batch CSV analysis**
  - **REST API endpoints** for integration :contentReference[oaicite:2]{index=2}

---

## 🧠 Approaches Used

The project experiments with three levels of intelligence (even though the Flask app currently uses a TextBlob-based engine):

1. **Baseline (Deployed in Flask app)**
   - Uses **TextBlob** polarity: `[-1, 1]`
   - Custom domain rules for smartwatch phrases  
     Examples:
     - “battery draining fast” → force **Negative**
     - “battery lasts all day” → force **Positive** 

2. **Classical ML Model (offline scripts)**
   - TF–IDF vectorizer + **Logistic Regression**
   - Trained on labelled smartwatch reviews
   - Evaluated with accuracy, precision, recall, F1-score :contentReference[oaicite:4]{index=4}

3. **Transformer Model (offline scripts)**
   - Fine-tuned **DistilBERT** for sentiment classification
   - Better at handling slang and contextual phrases than classical ML :contentReference[oaicite:5]{index=5}

> The Flask app (`app.py`) currently focuses on the **TextBlob + rules engine** and a rich UI for demonstrations. :contentReference[oaicite:6]{index=6}

---

## 🏗️ System Features

### 1. Authentication
- **Login / Signup** using SQLite (`auth.db`)
- Simple username + password system (for classroom demo only) 

### 2. Single Review Analysis ( `/` )
- Text area to paste one smartwatch review
- Optional checkbox: **“Proceed even if not clearly smartwatch-related”**
- Shows:
  - Smartwatch relevance badge
  - Overall sentiment (Positive / Neutral / Negative)
  - Polarity score (−1 to 1)
  - Confidence (%) = |polarity| × 100
  - Aspect-wise sentiment with evidence sentences 

### 3. Batch Analysis ( `/batch` )
- Upload CSV file (e.g., `smartwatch_genai.csv`)
- Auto-detect or specify text column
- For each review, compute:
  - Relevance
  - Sentiment
  - Polarity
  - Confidence
- Show:
  - Total rows, relevant vs irrelevant
  - Distribution of Positive / Neutral / Negative
  - Average polarity
  - First 200 processed rows in a scrollable table 

### 4. REST API Endpoints
Planned/implemented API routes (depending on version):

- `GET /api/health` – simple health check  
- `POST /api/analyze-review` – JSON in, sentiment out  
- `POST /api/analyze-batch` – CSV file upload → JSON summary & details :contentReference[oaicite:10]{index=10}

---

## 🗂️ Dataset

- File: **`smartwatch_genai.csv`** (1,597+ customer reviews) :contentReference[oaicite:11]{index=11}  
- Contains:
  - ~27 features (brand, category, dimensions, ratings, review text, etc.)
  - Suitable for:
    - Sentiment analysis
    - Opinion mining
    - Customer feedback analytics for smartwatch products 

---

## 🧱 Tech Stack

- **Backend:** Flask (Python)   
- **NLP:** TextBlob, classical ML (scikit-learn), Transformers (Hugging Face)   
- **Data Handling:** Pandas, NumPy  
- **ML:** scikit-learn, PyTorch, Transformers, Datasets  
- **Database:** SQLite (`auth.db`) for user accounts   
- **Frontend:** HTML + CSS (modern card-based UI)

Dependencies are listed in **`requirements.txt`**. :contentReference[oaicite:16]{index=16}

---

## 📁 Project Structure

> This is the intended structure; your repo may contain a subset of these files.

```text
.
├── app.py                     # Flask web application (routes + logic)
├── requirements.txt           # Python dependencies
├── auth.db                    # SQLite auth database (auto-created)
├── smartwatch_genai.csv       # Raw smartwatch reviews dataset (optional/demo)
├── data/
│   └── smartwatch_labelled.csv   # Cleaned & labelled dataset (offline training)
├── models/
│   ├── classical_model.pkl       # TF-IDF + Logistic Regression
│   └── transformer_distilbert/   # Fine-tuned DistilBERT
├── templates/                 # HTML templates (single review, batch, login, signup)
├── static/                    # CSS / assets for UI
├── GEN_AI.pptx                # Project presentation
└── Smartwatch Sentiment Analyzer Report.pdf   # Detailed project report
