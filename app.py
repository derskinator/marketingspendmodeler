import re
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from regression_model import clean_shopify_data, merge_spend_data, run_regression

# Streamlit page config
st.set_page_config(page_title="Spend vs. Direct Sales Regression", layout="wide")
st.title("Spend vs. Direct Sales Regression")

# Sidebar: CSV uploads
st.sidebar.header("Upload CSV Files")
shop_files = st.sidebar.file_uploader(
    "Shopify export CSVs", type="csv", accept_multiple_files=True
)
meta_file = st.sidebar.file_uploader("Meta spend CSV", type="csv")
google_file = st.sidebar.file_uploader("Google spend CSV", type="csv")

# Main logic: run when all files are provided
def load_and_prepare_data():
    # Shopify ingestion
    shop_dfs = []
    for f in shop_files:
        df = pd.read_csv(f)
        df.columns = df.columns.str.strip()
        m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
        if not m:
            st.error(f"Could not parse month from filename: {f.name}")
            st.stop()
        df['Month Start'] = pd.to_datetime(m.group(1))
        shop_dfs.append(df)
    shop_df = pd.concat(shop_dfs, ignore_index=True)
    direct_sales_df = clean_shopify_data(shop_df)

    # Meta spend ingestion
    meta_raw = pd.read_csv(meta_file)
    meta_raw.columns = meta_raw.columns.str.strip()
    if 'Month' not in meta_raw.columns or 'Amount spent (USD)' not in meta_raw.columns:
        st.error("Meta CSV must have 'Month' and 'Amount spent (USD)' columns.")
        st.stop()
    meta_raw['Month Start'] = pd.to_datetime(
        meta_raw['Month'].str.split(' - ').str[0], errors='coerce'
    )
    meta_clean = meta_raw.rename(columns={'Amount spent (USD)': 'Meta Spend'})[
        ['Month Start', 'Meta Spend']
    ]

    # Google spend ingestion
    google_raw = pd.read_csv(google_file, skiprows=2)
    google_raw.columns = google_raw.columns.str.strip()
    if 'Month' not in google_raw.columns or 'Cost' not in google_raw.columns:
        st.error("Google CSV must have 'Month' and 'Cost' columns.")
        st.stop()
    google_raw['Month Start'] = pd.to_datetime(
        google_raw['Month'], format='%B %Y', errors='coerce'
    )
    google_clean = google_raw.rename(columns={'Cost': 'Google Spend'})[
        ['Month Start', 'Google Spend']
    ]

    # Merge all data
    merged = merge_spend_data(direct_sales_df, meta_clean, google_clean)
    return merged

if shop_files and meta_file and google_file:
    merged_df = load_and_prepare_data()

    # Regression
    model, forecast_fn = run_regression(merged_df)

    # Display merged data
    st.subheader("Merged Sales & Spend Data")
    st.dataframe(merged_df)

    # Regression results
    st.subheader("Regression Results by Channel")
    coefs = pd.DataFrame({
        'Coefficient': model.params,
        'p-value': model.pvalues
    })
    st.table(coefs)
    st.markdown(f"**R-squared:** {model.rsquared:.3f}")

    # Scenario forecasting: percent adjustments
    st.subheader("Scenario Forecast")
    st.markdown("Adjust the percentage of historical spend to see month-by-month impact on Direct Sales and see actual spend values.")
    meta_pct = st.slider("Meta Spend (% of historical)", 0, 200, 100)
    google_pct = st.slider("Google Spend (% of historical)", 0, 200, 100)
    if st.button("Run Scenario Forecast"):
        params = model.params
        scenario = merged_df[['Month Start', 'Meta Spend', 'Google Spend']].copy()
        # calculate scenario spend amounts
        scenario['Scenario Meta Spend'] = scenario['Meta Spend'] * meta_pct / 100
        scenario['Scenario Google Spend'] = scenario['Google Spend'] * google_pct / 100
        # calculate estimated sales
        scenario['Estimated Sales'] = (
            params['const']
            + params['Meta Spend'] * scenario['Scenario Meta Spend']
            + params['Google Spend'] * scenario['Scenario Google Spend']
        )
        # Chart of scenario estimated sales
        st.subheader("Scenario: Estimated Direct Sales by Month")
        st.line_chart(scenario.set_index('Month Start')['Estimated Sales'])
        # Table: Month, historical spend, scenario spend, estimated sales
        scenario['Month'] = scenario['Month Start'].dt.strftime('%Y-%m')
        breakdown = scenario.set_index('Month')[[
            'Meta Spend', 'Scenario Meta Spend',
            'Google Spend', 'Scenario Google Spend',
            'Estimated Sales'
        ]]
        st.subheader("Month-by-Month Breakdown")
        st.dataframe(breakdown)

    # Historical vs fitted
    st.subheader("Historical vs Fitted Direct Sales")
    X = sm.add_constant(merged_df[['Meta Spend', 'Google Spend']])
    merged_df['Fitted'] = model.predict(X)
    chart_df = merged_df.set_index('Month Start')[['Direct Sales', 'Fitted']]
    st.line_chart(chart_df)
else:
    st.info("Please upload Shopify, Meta, and Google CSVs to run the analysis.")

