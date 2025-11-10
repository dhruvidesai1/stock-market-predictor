# Stock Predictor (GA + ANN) — Django App

**Python:** 3.12.4

This project is a minimal Django application that demonstrates:
- Genetic Algorithm (DEAP) optimizing ANN/LSTM initial weights
- Optional fine-tuning with backpropagation
- Frontend using Tailwind (CDN) and Chart.js with zoom/pan plugin
- SQLite used by Django for admin/auth (no persistent prediction storage required)

## What the SQLite DB stores
The included SQLite database (db.sqlite3 after migrations) is only used by Django's default apps (auth, admin, sessions).
Prediction results are not stored persistently in this version — predictions are computed on request.
You can extend the app to save results by adding Django models.

## Quick start
1. Create a virtualenv with Python 3.12.4 and activate it.
2. Install requirements:
   ```
   pip install -r requirements.txt
   ```
3. Run migrations and start the dev server:
   ```
   python manage.py migrate
   python manage.py runserver
   ```
4. Open http://127.0.0.1:8000/ and try predicting (e.g., AAPL, 2020-01-01 to 2024-01-01).

## Notes
- GA is computationally expensive. Defaults are conservative for development.
- Tailwind is included via CDN for simplicity — for production consider building Tailwind properly.
