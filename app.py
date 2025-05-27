import pandas as pd
import statsmodels.api as sm
import numpy as np


def clean_shopify_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw Shopify export and returns a DataFrame with monthly total sales (all channels).
    Expects df has columns including:
      - 'total_sales' or 'Total sales'
      - 'Month Start' (datetime)
    Aggregates and standardizes column names.
    """
    df = df.copy()
    # Normalize total sales column name
    if 'Total Sales' in df.columns:
        sales_col = 'Total Sales'
    elif 'Total sales' in df.columns:
        sales_col = 'Total sales'
    else:
        raise ValueError("DataFrame must contain a 'Total sales' or 'Total Sales' column.")

    if 'Month Start' not in df.columns:
        raise ValueError("DataFrame must contain 'Month Start' datetime column.")

    agg = (
        df.groupby('Month Start')[sales_col]
        .sum()
        .rename('Total Sales')
        .reset_index()
    )
    return agg


def merge_spend_data(sales_df: pd.DataFrame,
                     meta_df: pd.DataFrame,
                     google_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned Shopify Total Sales with Meta and Google spend on 'Month Start'.
    Expects:
      - sales_df with ['Month Start', 'Total Sales']
      - meta_df with ['Month Start', 'Meta Spend']
      - google_df with ['Month Start', 'Google Spend']
    """
    df = sales_df.copy()
    df = pd.merge(df, meta_df, on='Month Start', how='left')
    df = pd.merge(df, google_df, on='Month Start', how='left')
    df = df.sort_values('Month Start').reset_index(drop=True)
    return df


def run_regression(df: pd.DataFrame, transform: str = 'log'):
    """
    Runs OLS regression: Total Sales ~ Meta Spend + Google Spend.
    By default, applies log1p transform to tame skew.
    Returns the fitted statsmodels OLS results and a forecast function.

    forecast(meta_spend, google_spend) uses the same transform.
    """
    # Ensure required columns exist
    required = ['Total Sales', 'Meta Spend', 'Google Spend']
    if not all(col in df.columns for col in required):
        missing = [col for col in required if col not in df.columns]
        raise ValueError(f"DataFrame missing columns: {missing}")

    df_clean = df.dropna(subset=required).copy()

    # Apply transformation
    if transform == 'log':
        df_clean['X_meta'] = np.log1p(df_clean['Meta Spend'])
        df_clean['X_google'] = np.log1p(df_clean['Google Spend'])
    else:
        df_clean['X_meta'] = df_clean['Meta Spend']
        df_clean['X_google'] = df_clean['Google Spend']

    X = df_clean[['X_meta', 'X_google']]
    X = sm.add_constant(X)
    y = df_clean['Total Sales']

    model = sm.OLS(y, X).fit()

    def forecast(meta_spend: float, google_spend: float) -> float:
        # Build input row
        if transform == 'log':
            x_meta = np.log1p(meta_spend)
            x_google = np.log1p(google_spend)
        else:
            x_meta = meta_spend
            x_google = google_spend
        data = {'const': 1, 'X_meta': x_meta, 'X_google': x_google}
        return float(model.predict(pd.DataFrame([data]))[0])

    return model, forecast



