install:
	pip install -r requirements.txt
	cd frontend && npm install

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

train-vision:
	python -m src.pipelines.train_vision

train-recommender:
	python -m src.pipelines.train_recommender

serve:
	docker compose up --build

stop:
	docker compose down

serve-local-api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-local-model:
	mlflow models serve -m models/heritage_classifier --port 5001 --no-conda

serve-local-ui:
	cd frontend && npm run dev

test:
	export PYTHONPATH=$PYTHONPATH:$$(pwd) && pytest tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
