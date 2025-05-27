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

# Main logic function
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
    meta_raw['Month Start'] = pd.to_datetime(
        meta_raw['Month'].str.split(' - ').str[0], errors='coerce'
    )
    meta_clean = meta_raw.rename(columns={'Amount spent (USD)': 'Meta Spend'})[['Month Start', 'Meta Spend']]

    # Google spend ingestion
    google_raw = pd.read_csv(google_file, skiprows=2)
    google_raw.columns = google_raw.columns.str.strip()
    google_raw['Month Start'] = pd.to_datetime(
        google_raw['Month'], format='%B %Y', errors='coerce'
    )
    google_clean = google_raw.rename(columns={'Cost': 'Google Spend'})[['Month Start', 'Google Spend']]

    # Merge
    merged = merge_spend_data(direct_sales_df, meta_clean, google_clean)
    return merged

# Run app when files present
if shop_files and meta_file and google_file:
    merged_df = load_and_prepare_data()

    # Regression with log transform
    model, _ = run_regression(merged_df, transform='log')

    # Display merged data formatted
    st.subheader("Merged Sales & Spend Data")
    merged_display = merged_df.copy()
    merged_display['Direct Sales'] = merged_display['Direct Sales'].map("${:,.2f}".format)
    merged_display['Meta Spend'] = merged_display['Meta Spend'].map("${:,.2f}".format)
    merged_display['Google Spend'] = merged_display['Google Spend'].map("${:,.2f}".format)
    st.dataframe(merged_display)

    # Regression results
    st.subheader("Regression Results by Channel")
    coefs = pd.DataFrame({'Coefficient': model.params, 'p-value': model.pvalues})
    coefs_display = coefs.copy()
    coefs_display['Coefficient'] = coefs_display['Coefficient'].map("${:,.2f}".format)
    coefs_display['p-value'] = coefs_display['p-value'].map("{:.3f}".format)
    st.table(coefs_display)
    st.markdown(f"**R-squared:** {model.rsquared:.3f}")
    # Plain-language summary
    mean_meta = merged_df['Meta Spend'].mean()
    mean_google = merged_df['Google Spend'].mean()
    # Calculate approx marginal effect at average spend
    beta_meta = model.params['X_meta']
    beta_google = model.params['X_google']
    eff_meta = beta_meta / (mean_meta + 1)
    eff_google = beta_google / (mean_google + 1)
    st.markdown(f"- At average monthly Meta spend of ${mean_meta:,.2f}, each additional $1 is associated with approximately ${eff_meta:,.2f} increase in Direct Sales.")
    st.markdown(f"- At average monthly Google spend of ${mean_google:,.2f}, each additional $1 is associated with approximately ${eff_google:,.2f} increase in Direct Sales.")

    # Scenario forecasting
    st.subheader("Scenario Forecast")
    meta_pct = st.slider("Meta Spend (% of historical)", 0, 200, 100)
    google_pct = st.slider("Google Spend (% of historical)", 0, 200, 100)
    if st.button("Run Scenario Forecast"):
        params = model.params
        scenario = merged_df[['Month Start', 'Meta Spend', 'Google Spend']].copy()
        scenario['Scenario Meta Spend'] = scenario['Meta Spend'] * meta_pct / 100
        scenario['Scenario Google Spend'] = scenario['Google Spend'] * google_pct / 100
        scenario['Estimated Sales'] = (
            params['const']
            + params['X_meta'] * np.log1p(scenario['Scenario Meta Spend'])
            + params['X_google'] * np.log1p(scenario['Scenario Google Spend'])
        )
        # Format
        scenario['Meta Spend'] = scenario['Meta Spend'].map("${:,.2f}".format)
        scenario['Scenario Meta Spend'] = scenario['Scenario Meta Spend'].map("${:,.2f}".format)
        scenario['Google Spend'] = scenario['Google Spend'].map("${:,.2f}".format)
        scenario['Scenario Google Spend'] = scenario['Scenario Google Spend'].map("${:,.2f}".format)
        scenario['Estimated Sales'] = scenario['Estimated Sales'].map("${:,.2f}".format)
        # Chart and table
        st.line_chart(scenario.set_index('Month Start')['Estimated Sales'].str.replace('[$,]', '', regex=True).astype(float))
        st.subheader("Month-by-Month Breakdown")
        scenario['Month'] = scenario['Month Start'].dt.strftime('%Y-%m')
        st.dataframe(
            scenario.set_index('Month')[[
                'Meta Spend', 'Scenario Meta Spend',
                'Google Spend', 'Scenario Google Spend',
                'Estimated Sales'
            ]]
        )

    # Historical vs fitted
    st.subheader("Historical vs Fitted Direct Sales")
    X = sm.add_constant(merged_df[['Meta Spend', 'Google Spend']])
    merged_df['Fitted'] = model.predict(X)
    merged_df['Direct Sales'] = merged_df['Direct Sales'].map("${:,.2f}".format)
    merged_df['Fitted'] = merged_df['Fitted'].map("${:,.2f}".format)
    hist_display = merged_df.set_index('Month Start')[['Direct Sales', 'Fitted']]
    chart_df = hist_display.replace('[\$,]', '', regex=True).astype(float)
    st.line_chart(chart_df)

else:
    st.info("Please upload Shopify, Meta, and Google CSVs to run the analysis.")

