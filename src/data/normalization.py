import pandas as pd

NULL_LIKE_VALUES = {
    "", " ", "nan", "NaN", "none", "None", "null", "NULL", "N/A", "n/a"
}

def normalize_string(value):
    if pd.isna(value):
        return None
    value = str(value).strip()
    return None if value in NULL_LIKE_VALUES else value

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        if column != "Campaign_ID" and df[column].dtype == "object":
            df[column] = df[column].map(normalize_string)

    df["Campaign_ID"] = df["Campaign_ID"].astype("string").str.strip()

    for column in ["Duration", "Clicks", "Impressions", "Engagement_Score"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype("int64")

    for column in ["Conversion_Rate", "Acquisition_Cost", "ROI"]:
        df[column] = pd.to_numeric(df[column], errors="raise").astype(float)

    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    return df
