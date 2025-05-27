import re
import pandas as pd
import streamlit as st
import statsmodels.api as sm
from regression_model import clean_shopify_data, merge_spend_data, run_regression

st.set_page_config(page_title="Spend vs. Direct Sales Regression", layout="wide")
st.title("Spend vs. Direct Sales Regression")

# Sidebar: File Uploads
st.sidebar.header("Upload CSV Files")
shop_files = st.sidebar.file_uploader(
    "Upload Shopify export CSVs", type="csv", accept_multiple_files=True
)
meta_file = st.sidebar.file_uploader("Upload Meta spend CSV", type="csv")
google_file = st.sidebar.file_uploader("Upload Google spend CSV", type="csv")

if shop_files and meta_file and google_file:
    # --- Shopify Data Ingestion ---
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

    # --- Meta Spend Ingestion ---
    meta_raw = pd.read_csv(meta_file)
    meta_raw.columns = meta_raw.columns.str.strip()
    meta_raw['Month Start'] = pd.to_datetime(
        meta_raw['Month'].str.split(' - ').str[0], errors='coerce'
    )
    meta_clean = meta_raw.rename(columns={'Amount spent (USD)': 'Meta Spend'})[
        ['Month Start', 'Meta Spend']
    ]

    # --- Google Spend Ingestion ---
    google_raw = pd.read_csv(google_file, skiprows=2)
    google_raw.columns = google_raw.columns.str.strip()
    google_raw['Month Start'] = pd.to_datetime(
        google_raw['Month'], format='%B %Y', errors='coerce'
    )
    google_clean = google_raw.rename(columns={'Cost': 'Google Spend'})[
        ['Month Start', 'Google Spend']
    ]

    # --- Merge & Regression ---
    merged_df = merge_spend_data(direct_sales_df, meta_clean, google_clean)
    model, forecast_fn = run_regression(merged_df)

    # --- Display Merged Data ---
    st.subheader("Merged Sales & Spend Data")
    st.dataframe(merged_df)

    # --- Regression Results by Channel ---
    st.subheader("Regression Results by Channel")
    coefs = pd.DataFrame({
        'Coefficient': model.params,
        'p-value': model.pvalues
    })
    st.table(coefs)
    st.markdown(f"**R-squared:** {model.rsquared:.3f}")
    st.markdown(
        f"- A $1 increase in Meta spend is associated with **${model.params['Meta Spend']:.2f}** change in Direct Sales (p={model.pvalues['Meta Spend']:.3f})."
    )
    st.markdown(
        f"- A $1 increase in Google spend is associated with **${model.params['Google Spend']:.2f}** change in Direct Sales (p={model.pvalues['Google Spend']:.3f})."
    )

    # --- Forecast UI ---
    st.subheader("Forecast Direct Sales")
    with st.form("forecast_form"):
        meta_min, meta_max = merged_df['Meta Spend'].min(), merged_df['Meta Spend'].max()
        google_min, google_max = merged_df['Google Spend'].min(), merged_df['Google Spend'].max()
        meta_input = st.slider("Meta Spend", float(meta_min), float(meta_max), float((meta_min+meta_max)/2))
        google_input = st.slider("Google Spend", float(google_min), float(google_max), float((google_min+google_max)/2))
        submit = st.form_submit_button("Estimate Sales")

    if submit:
        prediction = forecast_fn(meta_input, google_input)
        st.metric(label="Predicted Direct Sales", value=f"${prediction:,.2f}")

        # --- Estimated Sales by Month ---
        st.subheader("Estimated Direct Sales by Month")
        est_df = merged_df[['Month Start']].copy()
        est_df['Estimated Sales'] = est_df.apply(
            lambda row: forecast_fn(meta_input, google_input), axis=1
        )
        # Line chart of estimated sales
        st.line_chart(est_df.set_index('Month Start')['Estimated Sales'])
        # Month-by-month breakdown table
        est_df['Month'] = est_df['Month Start'].dt.strftime('%Y-%m')
        st.dataframe(est_df[['Month', 'Estimated Sales']].set_index('Month'))

    # --- Historical vs Fitted Chart ---
    st.subheader("Historical vs Fitted Direct Sales")
    X = merged_df[['Meta Spend', 'Google Spend']]
    X = sm.add_constant(X)
    merged_df['Fitted'] = model.predict(X)
    chart_df = merged_df.set_index('Month Start')[['Direct Sales', 'Fitted']]
    st.line_chart(chart_df)

else:
    st.info("Please upload Shopify exports, Meta spend CSV, and Google spend CSV to run the analysis.")
