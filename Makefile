quality_checks:
	isort .
	black .
	pylint --recursive=y .

unit_tests:
	pytest tests/

prediction: quality_checks unit_tests
	python car_price_predictor.py