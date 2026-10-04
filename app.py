import os
from typing import Optional

import pandas as pd
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "movie_metadata.csv")
MODEL = None
LABEL_ENCODER = None
GENRES = []
MODEL_ERROR = ""


def load_model():
    global MODEL, LABEL_ENCODER, GENRES, MODEL_ERROR

    if not os.path.exists(DATASET_PATH):
        MODEL = None
        LABEL_ENCODER = None
        GENRES = []
        MODEL_ERROR = "Dataset not found. Please place movie_metadata.csv in the project folder and refresh the page."
        return

    try:
        df = pd.read_csv(DATASET_PATH)
        required_cols = ["budget", "duration", "genres", "num_voted_users", "title_year", "imdb_score"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        df = df[required_cols].dropna().copy()
        df["main_genre"] = df["genres"].fillna("").astype(str).apply(lambda x: x.split("|")[0])
        df["hit"] = (df["imdb_score"] >= 7).astype(int)

        model_df = df.drop(columns=["genres", "imdb_score"]).copy()
        le = LabelEncoder()
        model_df["main_genre"] = le.fit_transform(model_df["main_genre"])
        X = model_df.drop(columns=["hit"])
        y = model_df["hit"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X_train, y_train)

        MODEL = clf
        LABEL_ENCODER = le
        GENRES = sorted(df["main_genre"].unique().tolist())
        MODEL_ERROR = ""
    except Exception as exc:  # pragma: no cover - protects page rendering in missing-data cases
        MODEL = None
        LABEL_ENCODER = None
        GENRES = []
        MODEL_ERROR = f"Unable to load the model: {exc}"


@app.route("/", methods=["GET", "POST"])
def index():
    if MODEL is None:
        load_model()

    result = None
    form_data = {}

    if request.method == "POST":
        form_data = {
            "budget": request.form.get("budget", ""),
            "duration": request.form.get("duration", ""),
            "genre": request.form.get("genre", ""),
            "num_voted_users": request.form.get("num_voted_users", ""),
            "title_year": request.form.get("title_year", ""),
        }

        if MODEL is not None:
            try:
                budget = float(form_data["budget"])
                duration = float(form_data["duration"])
                votes = float(form_data["num_voted_users"])
                year = float(form_data["title_year"])
                genre = form_data["genre"]

                if not genre:
                    raise ValueError("Please choose a genre.")

                if LABEL_ENCODER is None or genre not in LABEL_ENCODER.classes_:
                    raise ValueError("Selected genre is not available in the dataset.")

                encoded_genre = int(LABEL_ENCODER.transform([genre])[0])
                feature_row = [[budget, duration, encoded_genre, votes, year]]
                prediction = MODEL.predict(feature_row)[0]
                probabilities = MODEL.predict_proba(feature_row)[0]
                confidence = max(probabilities)
                hit_label = "Hit" if prediction == 1 else "Not a Hit"
                result = {
                    "prediction": hit_label,
                    "confidence": round(float(confidence) * 100, 2),
                    "probability_hit": round(float(probabilities[1]) * 100, 2) if len(probabilities) > 1 else 0,
                    "probability_not_hit": round(float(probabilities[0]) * 100, 2) if len(probabilities) > 1 else 0,
                }
            except ValueError as exc:
                result = {"error": str(exc)}

    return render_template(
        "index.html",
        result=result,
        genres=GENRES,
        model_error=MODEL_ERROR,
        form_data=form_data,
    )


if __name__ == "__main__":
    load_model()
    app.run(debug=True)
