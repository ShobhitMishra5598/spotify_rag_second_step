import pandas as pd
from src.data.document_builder import build_campaign_document

def test_campaign_document():
    row = pd.Series({
        "Campaign_ID": "C000001", "Company": "Nike",
        "Target_Audience": "Women 25-34", "Campaign_Goal": "Market Expansion",
        "Duration": 59, "Channel_Used": "Twitter", "Conversion_Rate": 21.47,
        "Acquisition_Cost": 4354.64, "ROI": 1.11, "Location": "Los Angeles",
        "Language": "French", "Clicks": 8339, "Impressions": 405363,
        "Engagement_Score": 5, "Customer_Segment": "Foodies",
        "Date": pd.Timestamp("2025-01-01"),
    })
    document = build_campaign_document(row)
    assert document.metadata["campaign_id"] == "C000001"
    assert "Company: Nike" in document.page_content
