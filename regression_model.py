# regression_model.py

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
    return df.sort_values('Month Start').reset_index(drop=True)

def run_regression(df: pd.DataFrame, transform: str = 'log') -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Runs OLS regression: Total Sales ~ Meta Spend + Google Spend.
    By default, applies log1p transform to tame skew.
    Returns the fitted statsmodels OLS results.
    """
    # Validate
    required = ['Total Sales', 'Meta Spend', 'Google Spend']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")

    dfc = df.dropna(subset=required).copy()

    if transform == 'log':
        dfc['X_meta']   = np.log1p(dfc['Meta Spend'])
        dfc['X_google'] = np.log1p(dfc['Google Spend'])
    else:
        dfc['X_meta']   = dfc['Meta Spend']
        dfc['X_google'] = dfc['Google Spend']

    X = sm.add_constant(dfc[['X_meta','X_google']])
    y = dfc['Total Sales']
    model = sm.OLS(y, X).fit()
    return model
