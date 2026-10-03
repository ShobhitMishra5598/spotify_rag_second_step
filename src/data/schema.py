from typing import Final

REQUIRED_COLUMNS: Final = [
    "Campaign_ID", "Company", "Target_Audience", "Campaign_Goal",
    "Duration", "Channel_Used", "Conversion_Rate", "Acquisition_Cost",
    "ROI", "Location", "Language", "Clicks", "Impressions",
    "Engagement_Score", "Customer_Segment", "Date",
]

STRING_COLUMNS: Final = [
    "Campaign_ID", "Company", "Target_Audience", "Campaign_Goal",
    "Channel_Used", "Location", "Language", "Customer_Segment",
]

INTEGER_COLUMNS: Final = ["Duration", "Clicks", "Impressions", "Engagement_Score"]
FLOAT_COLUMNS: Final = ["Conversion_Rate", "Acquisition_Cost", "ROI"]
