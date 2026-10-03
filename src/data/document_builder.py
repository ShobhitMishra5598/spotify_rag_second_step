import pandas as pd
from langchain_core.documents import Document

def _display(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return str(value)

def build_campaign_document(row: pd.Series) -> Document:
    campaign_id = _display(row["Campaign_ID"])

    text = f"""Campaign ID: {campaign_id}
Company: {_display(row["Company"])}
Target Audience: {_display(row["Target_Audience"])}
Campaign Goal: {_display(row["Campaign_Goal"])}
Duration: {_display(row["Duration"])} days
Channel Used: {_display(row["Channel_Used"])}
Conversion Rate: {_display(row["Conversion_Rate"])}%
Acquisition Cost: {_display(row["Acquisition_Cost"])}
ROI: {_display(row["ROI"])}
Location: {_display(row["Location"])}
Language: {_display(row["Language"])}
Clicks: {_display(row["Clicks"])}
Impressions: {_display(row["Impressions"])}
Engagement Score: {_display(row["Engagement_Score"])}
Customer Segment: {_display(row["Customer_Segment"])}
Date: {_display(row["Date"].date())}"""

    metadata = {
        "campaign_id": campaign_id,
        "company": row["Company"],
        "target_audience": row["Target_Audience"],
        "campaign_goal": row["Campaign_Goal"],
        "duration": int(row["Duration"]),
        "channel_used": row["Channel_Used"],
        "conversion_rate": float(row["Conversion_Rate"]),
        "acquisition_cost": float(row["Acquisition_Cost"]),
        "roi": float(row["ROI"]),
        "location": row["Location"],
        "language": None if pd.isna(row["Language"]) else row["Language"],
        "clicks": int(row["Clicks"]),
        "impressions": int(row["Impressions"]),
        "engagement_score": int(row["Engagement_Score"]),
        "customer_segment": row["Customer_Segment"],
        "date": str(row["Date"].date()),
    }

    return Document(page_content=text, metadata=metadata)
