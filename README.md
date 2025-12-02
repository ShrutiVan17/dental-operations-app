# Dental Prediction + LLM Strategy Agent (VS Code Ready)

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # add your OpenAI key
```
## Data
Your file is placed at `data/datasetcleaned.csv`.

Detected columns:
- date: **DateKpi**
- no-show: **VisitsNoShow**
- hygiene: **VisitsHygieneCompleted**
- restorative: **VisitsRestorativeCompleted**
- collections: **Collections**
- profit: **None**

## Run
```bash
streamlit run app.py
```
