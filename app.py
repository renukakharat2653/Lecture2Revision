
import os
import re
from flask import Flask, render_template, request, jsonify
from pypdf import PdfReader
import torch
import torch.nn as nn
from transformers import pipeline

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
os.makedirs("uploads", exist_ok=True)

summarizer = None


class ScoreLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


def load_lstm():
    path = "models/lstm_model.pth"
    if not os.path.exists(path):
        return None

    model = ScoreLSTM()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def read_upload(file):
    name = file.filename.lower()

    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        reader = PdfReader(file)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    raise ValueError("Please upload a TXT or PDF file.")


def summarize_text(text):
    global summarizer

    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return "No readable text found. Please paste or upload a lecture."

    # Transformer summarization
    try:
        if summarizer is None:
            summarizer = pipeline(
                "summarization",
                model="sshleifer/distilbart-cnn-12-6"
            )

        chunk = clean[:5000]
        result = summarizer(
            chunk,
            max_length=130,
            min_length=25,
            do_sample=False,
            truncation=True
        )
        return result[0]["summary_text"]

    except Exception as exc:
        print("Transformer unavailable; using extractive fallback:", exc)
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        return " ".join(sentences[:5])


def make_notes(summary):
    sentences = re.split(r"(?<=[.!?])\s+", summary)
    return [s.strip() for s in sentences if s.strip()]


def make_quiz(text):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    questions = []

    for sentence in sentences:
        words = re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{4,}\b", sentence)
        if len(words) < 2:
            continue

        answer = max(words, key=len)
        question = re.sub(
            r"\b" + re.escape(answer) + r"\b",
            "________",
            sentence,
            count=1
        )

        questions.append({
            "question": question,
            "answer": answer
        })

        if len(questions) == 5:
            break

    return questions


@app.route("/", methods=["GET", "POST"])
def home():
    lecture_text = ""
    summary = ""
    notes = []
    quiz = []
    error = ""

    if request.method == "POST":
        lecture_text = request.form.get("lecture_text", "").strip()
        uploaded = request.files.get("lecture_file")

        try:
            if uploaded and uploaded.filename:
                lecture_text = read_upload(uploaded).strip()

            if not lecture_text:
                raise ValueError("Paste lecture text or upload a PDF/TXT file.")

            summary = summarize_text(lecture_text)
            notes = make_notes(summary)
            quiz = make_quiz(lecture_text)

        except Exception as exc:
            error = str(exc)

    return render_template(
        "index.html",
        lecture_text=lecture_text,
        summary=summary,
        notes=notes,
        quiz=quiz,
        error=error
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    try:
        scores = [float(s) for s in data.get("scores", [])]
        if len(scores) != 5 or any(s < 0 or s > 100 for s in scores):
            return jsonify({
                "error": "Enter exactly 5 scores from 0 to 100."
            }), 400

        model = load_lstm()
        if model is None:
            return jsonify({
                "error": "LSTM model missing. Run: python train_model.py"
            }), 400

        x = torch.tensor(scores, dtype=torch.float32).view(1, 5, 1) / 100.0

        with torch.no_grad():
            result = model(x)
            needs_revision = int(result.argmax(dim=1).item())

        if needs_revision:
            advice = "Focus on revision. Review weak topics and retry a short quiz."
        else:
            advice = "Your recent scores look stronger. Practise mixed questions to maintain progress."

        return jsonify({
            "needs_revision": needs_revision,
            "advice": advice,
            "note": "Demo model trained on synthetic scores; not a validated student assessment."
        })

    except ValueError:
        return jsonify({"error": "Scores must be numeric."}), 400


if __name__ == "__main__":
    app.run(debug=True)
