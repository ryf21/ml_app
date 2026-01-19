\# ML FastAPI Docker Lab



Endpoints:

\- GET /health

\- POST /predict {"text": "..."}

\- POST /predict\_batch {"texts": \["a","b"]}

\- GET /model\_info



Run locally:

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload



Run tests:

pytest -q



Build docker:

docker build -t ml\_app:latest .



Run docker:

docker run --rm -p 8000:8000 ml\_app:latest



