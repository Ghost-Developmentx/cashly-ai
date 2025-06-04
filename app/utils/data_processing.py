import pandas as pd
import re
from sklearn.preprocessing import StandardScaler


def clean_transaction_description(description):
    """Cleans and normalizes transaction descriptions"""
    if not description:
        return ""

    # Convert to lowercase
    desc = description.lower()

    # Remove common transaction prefixes/suffixes
    desc = re.sub(
        r"(payment to|payment from|purchase at|tx-|pos |debit card purchase |credit card |check |ach |deposit |withdrawal )",
        "",
        desc,
    )

    # Remove special characters and extra spaces
    desc = re.sub(r"[^\w\s]", " ", desc)
    desc = re.sub(r"\s+", " ", desc).strip()

    return desc



def extract_transaction_features(transactions_df):
    """
    Extract features from transaction data for ML models

    Args:
        transactions_df: DataFrame with columns [date, amount, description]

    Returns:
        DataFrame with extracted features
    """
    df = transactions_df.copy()

    # Basic features
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["day_of_month"] = pd.to_datetime(df["date"]).dt.day
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

    # Amount features
    df["amount_abs"] = df["amount"].abs()
    df["is_expense"] = (df["amount"] < 0).astype(int)
    df["is_income"] = (df["amount"] > 0).astype(int)

    # Description features
    if "description" in df.columns:
        df["clean_description"] = df["description"].apply(clean_transaction_description)

        # Extract word count
        df["description_word_count"] = df["clean_description"].apply(
            lambda x: len(x.split()) if x else 0
        )

        # Check for common indicators
        df["is_subscription"] = (
            df["clean_description"]
            .str.contains(
                "subscription|monthly|netflix|spotify|amazon prime|hulu|disney"
            )
            .astype(int)
        )
        df["is_food"] = (
            df["clean_description"]
            .str.contains(
                "restaurant|food|grocery|pizza|burger|cafe|coffee|doordash|ubereats|grubhub"
            )
            .astype(int)
        )
        df["is_transportation"] = (
            df["clean_description"]
            .str.contains(
                "uber|lyft|taxi|parking|gas|fuel|transit|transport|train|bus|subway"
            )
            .astype(int)
        )
        df["is_shopping"] = (
            df["clean_description"]
            .str.contains("amazon|walmart|target|shop|store|retail|purchase|buy")
            .astype(int)
        )

    return df


def prepare_timeseries_data(transactions_df, freq="D", fill_method="ffill"):
    """
    Prepare transaction data for time series analysis

    Args:
        transactions_df: DataFrame with 'date' and 'amount' columns
        freq: Frequency for resampling ('D' for daily, 'W' for weekly, 'M' for monthly)
        fill_method: Method for filling missing values

    Returns:
        DataFrame resampled to specified frequency
    """
    # Convert to datetime if not already
    df = transactions_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")

    # Group by date and sum amounts
    daily_amounts = df["amount"].resample(freq).sum()

    # Fill missing values using pandas 2.x methods
    if fill_method == "ffill":
        daily_amounts = daily_amounts.ffill()
    elif fill_method == "bfill":
        daily_amounts = daily_amounts.bfill()
    else:
        daily_amounts = daily_amounts.fillna(0)

    # Reset index to get date as a column
    daily_amounts = daily_amounts.reset_index()

    return daily_amounts


def normalize_features(df, columns_to_normalize=None, return_scaler=False):
    """
    Normalize numerical features to improve model performance

    Args:
        df: DataFrame with features
        columns_to_normalize: List of columns to normalize (defaults to all numeric)
        return_scaler: Whether to return the scaler object for later use

    Returns:
        DataFrame with normalized features and optionally the scaler
    """
    df_norm = df.copy()

    # If no columns specified, normalize all numeric columns
    if not columns_to_normalize:
        numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
        columns_to_normalize = [col for col in numeric_cols]

    # Create scaler
    scaler = StandardScaler()

    # Fit and transform
    df_norm[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

    if return_scaler:
        return df_norm, scaler

    return df_norm
