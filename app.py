import pandas as pd
import statsmodels.api as sm
import numpy as np


def clean_shopify_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw Shopify export and returns a DataFrame with monthly total sales (all channels).
    Expects df has columns including:
      - 'Total sales'
      - 'Month Start' (datetime)
    """
    df = df.copy()

    if 'Month Start' not in df.columns:
        raise ValueError("DataFrame must contain 'Month Start' datetime column.")

    # Aggregate all sales by month
    agg = (
        df.groupby('Month Start')['Total sales']
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
    Returns the fitted results and a forecast function.
    """
    df_clean = df.dropna(subset=['Total Sales', 'Meta Spend', 'Google Spend']).copy()

    if transform == 'log':
        df_clean['X_meta'] = np.log1p(df_clean['Meta Spend'])
        df_clean['X_google'] = np.log1p(df_clean['Google Spend'])
        var_meta = 'X_meta'
        var_google = 'X_google'
    else:
        df_clean['X_meta'] = df_clean['Meta Spend']
        df_clean['X_google'] = df_clean['Google Spend']
        var_meta = 'X_meta'
        var_google = 'X_google'

    X = df_clean[[var_meta, var_google]]
    X = sm.add_constant(X)
    y = df_clean['Total Sales']

    model = sm.OLS(y, X).fit()

    def forecast(meta_spend: float, google_spend: float) -> float:
        data = {'const': 1}
        if transform == 'log':
            data[var_meta] = np.log1p(meta_spend)
            data[var_google] = np.log1p(google_spend)
        else:
            data[var_meta] = meta_spend
            data[var_google] = google_spend
        return float(model.predict(pd.DataFrame([data]))[0])

    return model, forecast


