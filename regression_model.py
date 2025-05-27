import pandas as pd
import statsmodels.api as sm

def clean_shopify_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw Shopify export and returns a DataFrame with monthly total Direct sales.
    Treats missing or 'direct' Referring channel as Direct sales.
    Expects df has columns including:
      - 'Referring channel'
      - 'Total sales'
      - 'Month Start' (datetime)
    """
    df = df.copy()
    # Treat missing or None 'Referring channel' as Direct
    direct_mask = df['Referring channel'].isna() | (df['Referring channel'] == 'direct')
    df['Sale Source'] = None
    df.loc[direct_mask, 'Sale Source'] = 'Direct'

    if 'Month Start' not in df.columns:
        raise ValueError("DataFrame must contain 'Month Start' datetime column.")

    # Aggregate Direct sales by month
    agg = (
        df[df['Sale Source'] == 'Direct']
        .groupby('Month Start')['Total sales']
        .sum()
        .rename('Direct Sales')
        .reset_index()
    )
    return agg

def merge_spend_data(shop_df: pd.DataFrame,
                     meta_df: pd.DataFrame,
                     google_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned Shopify Direct sales with Meta and Google spend on 'Month Start'.
    Expects:
      - shop_df with ['Month Start', 'Direct Sales']
      - meta_df with ['Month Start', 'Meta Spend']
      - google_df with ['Month Start', 'Google Spend']
    """
    df = shop_df.copy()
    df = pd.merge(df, meta_df, on='Month Start', how='left')
    df = pd.merge(df, google_df, on='Month Start', how='left')
    df = df.sort_values('Month Start').reset_index(drop=True)
    return df

def run_regression(df: pd.DataFrame):
    """
    Runs OLS regression: Direct Sales ~ Meta Spend + Google Spend.
    Returns the fitted results and a forecast function.
    """
    # Drop rows with missing values in any key columns
    df_clean = df.dropna(subset=['Direct Sales', 'Meta Spend', 'Google Spend'])

    X = df_clean[['Meta Spend', 'Google Spend']]
    X = sm.add_constant(X)
    y = df_clean['Direct Sales']

    model = sm.OLS(y, X).fit()

    def forecast(meta_spend: float, google_spend: float) -> float:
        """
        Given spend inputs, returns predicted Direct Sales.
        """
        data = {'const': 1, 'Meta Spend': meta_spend, 'Google Spend': google_spend}
        return float(model.predict(pd.DataFrame([data]))[0])

    return model, forecast
