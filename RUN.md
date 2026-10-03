# Run

## 1. Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Prepare and validate campaign data

```bash
python -m src.ingestion.prepare_campaigns
```

## 4. Run tests

```bash
pytest -v
```

## 5. Run the existing RAG chatbot

Create `.env` with the API settings required by `rag_chatbot.py`, then run:

```bash
python rag_chatbot.py
```

Use prompts at the `You:` input.
