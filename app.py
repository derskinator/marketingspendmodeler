import re
import pandas as pd
import streamlit as st
import statsmodels.api as sm
import numpy as np
from regression_model import clean_shopify_data, merge_spend_data, run_regression

# Page setup
st.set_page_config(page_title="Spend vs. Total Sales Regression", layout="wide")
st.title("Spend vs. Total Sales Regression")

# Sidebar file uploads
st.sidebar.header("Upload CSV Files")
shop_files = st.sidebar.file_uploader(
    "Shopify export CSVs", type="csv", accept_multiple_files=True
)
meta_file = st.sidebar.file_uploader("Meta spend CSV", type="csv")
google_file = st.sidebar.file_uploader("Google spend CSV", type="csv")

# Data preparation function
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
    total_sales_df = clean_shopify_data(shop_df)

    # Meta spend ingestion
    meta_raw = pd.read_csv(meta_file)
    meta_raw.columns = meta_raw.columns.str.strip()
    if 'Month' not in meta_raw.columns or 'Amount spent (USD)' not in meta_raw.columns:
        st.error("Meta CSV must have 'Month' and 'Amount spent (USD)' columns.")
        st.stop()
    meta_raw['Month Start'] = pd.to_datetime(
        meta_raw['Month'].str.split(' - ').str[0], errors='coerce'
    )
    meta_clean = meta_raw.rename(columns={'Amount spent (USD)': 'Meta Spend'})[['Month Start','Meta Spend']]

    # Google spend ingestion
    google_raw = pd.read_csv(google_file, skiprows=2)
    google_raw.columns = google_raw.columns.str.strip()
    if 'Month' not in google_raw.columns or 'Cost' not in google_raw.columns:
        st.error("Google CSV must have 'Month' and 'Cost' columns.")
        st.stop()
    google_raw['Month Start'] = pd.to_datetime(
        google_raw['Month'], format='%B %Y', errors='coerce'
    )
    google_clean = google_raw.rename(columns={'Cost': 'Google Spend'})[['Month Start','Google Spend']]

    # Merge
    merged_df = merge_spend_data(total_sales_df, meta_clean, google_clean)
    return merged_df

# Main app logic
if shop_files and meta_file and google_file:
    merged_df = load_and_prepare_data()

    # Regression (log-transform)
    model = run_regression(merged_df, transform='log')

    # Display merged table
    st.subheader("Merged Total Sales & Spend Data")
    disp = merged_df.copy()
    disp['Total Sales'] = disp['Total Sales'].map("${:,.2f}".format)
    disp['Meta Spend']   = disp['Meta Spend'].map("${:,.2f}".format)
    disp['Google Spend'] = disp['Google Spend'].map("${:,.2f}".format)
    st.dataframe(disp)

    # Regression results
    st.subheader("Regression Results by Channel")
    coefs = pd.DataFrame({'Coefficient': model.params, 'p-value': model.pvalues})
    coefs_disp = coefs.copy()
    coefs_disp['Coefficient'] = coefs_disp['Coefficient'].map("${:,.2f}".format)
    coefs_disp['p-value']    = coefs_disp['p-value'].map("{:.3f}".format)
    st.table(coefs_disp)
    st.markdown(f"**R-squared:** {model.rsquared:.3f}")

    # Plain-language summary
    avg_meta   = merged_df['Meta Spend'].mean()
    avg_google = merged_df['Google Spend'].mean()
    beta_meta  = model.params['X_meta']
    beta_google= model.params['X_google']
    eff_meta   = beta_meta  / (avg_meta + 1)
    eff_google = beta_google/ (avg_google + 1)
    st.markdown(f"- At avg Meta spend ${avg_meta:,.2f}, each extra $1 yields ~${eff_meta:,.2f} in Total Sales.")
    st.markdown(f"- At avg Google spend ${avg_google:,.2f}, each extra $1 yields ~${eff_google:,.2f} in Total Sales.")

    # Scenario forecast
    st.subheader("Scenario Forecast")
    # Meta slider
    meta_pct = st.slider("Meta Spend (% of historical)", 0, 200, 100)
    meta_val = avg_meta * meta_pct / 100
    st.write(f"Meta Spend (scenario): **${meta_val:,.2f}**")
    # Google slider
    google_pct = st.slider("Google Spend (% of historical)", 0, 200, 100)
    google_val = avg_google * google_pct / 100
    st.write(f"Google Spend (scenario): **${google_val:,.2f}**")

    if st.button("Run Scenario Forecast"):
        params = model.params
        scenario = merged_df[['Month Start','Total Sales','Meta Spend','Google Spend']].copy()
        scenario['Scenario Meta Spend']   = scenario['Meta Spend']   * meta_pct/100
        scenario['Scenario Google Spend'] = scenario['Google Spend'] * google_pct/100
        # Predicted sales
        x_meta   = np.log1p(scenario['Scenario Meta Spend'])
        x_google = np.log1p(scenario['Scenario Google Spend'])
        X_pred   = sm.add_constant(pd.DataFrame({'X_meta': x_meta,'X_google': x_google}))
        scenario['Estimated Sales'] = model.predict(X_pred)
        # Calculate increases
        scenario['Dollar Increase'] = scenario['Estimated Sales'] - scenario['Total Sales']
        scenario['Pct Increase']    = 100 * scenario['Dollar Increase'] / scenario['Total Sales']
        # Format
        scenario['Total Sales'] = scenario['Total Sales'].map("${:,.2f}".format)
        for c in ['Meta Spend','Scenario Meta Spend','Google Spend','Scenario Google Spend','Estimated Sales','Dollar Increase']:
            scenario[c] = scenario[c].map("${:,.2f}".format)
        scenario['Pct Increase'] = scenario['Pct Increase'].map("{:.2f}%".format)

        # Chart
        st.line_chart(
            scenario.set_index('Month Start')['Estimated Sales']
              .str.replace('[$,]','',regex=True).astype(float)
        )
        scenario['Month'] = scenario['Month Start'].dt.strftime('%Y-%m')
        st.subheader("Month-by-Month Breakdown")
        st.dataframe(scenario.set_index('Month')[[
            'Total Sales','Scenario Meta Spend','Scenario Google Spend',
            'Estimated Sales','Dollar Increase','Pct Increase'
        ]])

else:
    st.info("Please upload Shopify, Meta, and Google spend CSVs to run the analysis.")

