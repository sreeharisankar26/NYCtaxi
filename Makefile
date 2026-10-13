.PHONY: all data test app
all:            ## full pipeline from raw files
	python run_all.py
data:           ## download raw files (bash; on Windows use scripts/download.ps1)
	bash scripts/download.sh
test:
	python -m pytest -q tests
app:
	streamlit run app/app.py
