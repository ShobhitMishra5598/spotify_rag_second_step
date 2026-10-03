import pandas as pd
import pytest
from src.data.validation import validate_campaign_ids

def test_duplicate_campaign_ids_are_rejected():
    df = pd.DataFrame({"Campaign_ID": ["C000001", "C000001"]})
    with pytest.raises(ValueError):
        validate_campaign_ids(df)
