import pandas as pd
from .schema import REQUIRED_COLUMNS, INTEGER_COLUMNS, FLOAT_COLUMNS

def validate_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

def validate_campaign_ids(df: pd.DataFrame) -> None:
    ids = df["Campaign_ID"]
    if ids.isna().any():
        raise ValueError("Campaign_ID contains missing values.")
    if ids.duplicated().any():
        duplicates = ids[ids.duplicated(keep=False)].drop_duplicates().tolist()
        raise ValueError(f"Duplicate Campaign_ID values: {duplicates}")

def validate_required_values(df: pd.DataFrame) -> None:
    # Language may be missing in the supplied dataset.
    for column in ["Company", "Target_Audience", "Campaign_Goal",
                   "Channel_Used", "Location", "Customer_Segment", "Date"]:
        if df[column].isna().any():
            raise ValueError(f"Required field '{column}' contains missing values.")

def validate_numeric_columns(df: pd.DataFrame) -> None:
    for column in INTEGER_COLUMNS + FLOAT_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise TypeError(f"{column} must be numeric.")

def validate_campaign_data(df: pd.DataFrame) -> None:
    validate_columns(df)
    validate_campaign_ids(df)
    validate_required_values(df)
    validate_numeric_columns(df)
