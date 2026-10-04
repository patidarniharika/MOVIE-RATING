# Movie Rating Prediction Web App

This project contains a small Flask web app that predicts whether a movie is a "Hit" or "Not a Hit" based on the same features used in the notebook: budget, runtime, genre, number of votes, and release year.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the web app:
   ```bash
   python app.py
   ```

3. Open the browser at:
   ```text
   http://127.0.0.1:5000/
   ```

## Files

- `app.py` — Flask app and model training logic
- `templates/index.html` — UI for the prediction form
- `static/style.css` — webpage styling
- `main.py` — original notebook-style Python script

## Notes

The model is trained from `movie_metadata.csv` when the app starts. If the dataset is missing, the page shows a clear warning.
