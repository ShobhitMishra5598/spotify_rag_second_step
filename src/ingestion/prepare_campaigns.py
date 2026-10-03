import json
from pathlib import Path
import pandas as pd

from src.data.document_builder import build_campaign_document
from src.data.normalization import normalize_dataframe
from src.data.validation import validate_campaign_data

RAW_FILE = Path("data/raw/campaigns.csv")
PROCESSED_DIR = Path("data/processed")

def prepare_campaigns():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_FILE, dtype={"Campaign_ID": "string"})
    validate_campaign_data(df)

    df = normalize_dataframe(df)
    validate_campaign_data(df)

    validated_csv = PROCESSED_DIR / "validated_campaigns.csv"
    df.to_csv(validated_csv, index=False)

    documents = [build_campaign_document(row) for _, row in df.iterrows()]

    documents_file = PROCESSED_DIR / "campaign_documents.jsonl"
    with documents_file.open("w", encoding="utf-8") as file:
        for document in documents:
            file.write(json.dumps({
                "page_content": document.page_content,
                "metadata": document.metadata,
            }, ensure_ascii=False, default=str) + "\n")

    report = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "unique_campaign_ids": int(df["Campaign_ID"].nunique()),
        "duplicate_campaign_ids": int(df["Campaign_ID"].duplicated().sum()),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items()},
    }

    (PROCESSED_DIR / "validation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    print(f"Prepared {len(documents):,} campaign documents.")
    print(f"Validated CSV: {validated_csv}")
    print(f"Documents: {documents_file}")
    print(f"Report: {PROCESSED_DIR / 'validation_report.json'}")

if __name__ == "__main__":
    prepare_campaigns()
