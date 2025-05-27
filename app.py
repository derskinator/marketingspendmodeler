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

if st.sidebar.button("Run Analysis"):
    # Validate uploads
    if not shop_files or not meta_file or not google_file:
        st.sidebar.error("Please upload all required files before running.")
    else:
        # ---- Shopify Data Ingestion ----
        shop_dfs = []
        for f in shop_files:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()

            # Parse month start from filename: look for YYYY-MM-DD
            m = re.search(r"(\d{4}-\d{2}-\d{2})", f.name)
            if not m:
                st.error(f"Could not parse month from filename: {f.name}")
                st.stop()
            month_start = pd.to_datetime(m.group(1))

            df["Month Start"] = month_start
            shop_dfs.append(df)

        shop_df = pd.concat(shop_dfs, ignore_index=True)
        direct_sales_df = clean_shopify_data(shop_df)

        # ---- Meta Spend Ingestion ----
        meta_raw = pd.read_csv(meta_file)
        meta_raw.columns = meta_raw.columns.str.strip()
        if "Month" not in meta_raw.columns:
            st.error("Meta CSV missing 'Month' column.")
            st.stop()
        # split "2023-06-01 - 2023-06-30"
        meta_raw["Month Start"] = (
            pd.to_datetime(meta_raw["Month"].str.split(" - ").str[0], errors="coerce")
        )
        if "Amount spent (USD)" in meta_raw.columns:
            meta_clean = meta_raw.rename(
                columns={"Amount spent (USD)": "Meta Spend"}
            )[
                ["Month Start", "Meta Spend"]
            ]
        else:
            st.error("Meta CSV missing 'Amount spent (USD)' column.")
            st.stop()

        # ---- Google Spend Ingestion ----
        # Skip the header rows if present
        google_raw = pd.read_csv(google_file, skiprows=2)
        google_raw.columns = google_raw.columns.str.strip()
        if "Month" not in google_raw.columns or "Cost" not in google_raw.columns:
            st.error("Google CSV must have 'Month' and 'Cost' columns.")
            st.stop()
        google_raw["Month Start"] = pd.to_datetime(
            google_raw["Month"], format="%B %Y", errors="coerce"
        )
        google_clean = google_raw.rename(columns={"Cost": "Google Spend"})[
            ["Month Start", "Google Spend"]
        ]

        # ---- Merge & Display ----
        merged_df = merge_spend_data(direct_sales_df, meta_clean, google_clean)
        st.subheader("Merged Sales & Spend Data")
        st.dataframe(merged_df)

        # ---- Regression ----
        model, forecast_fn = run_regression(merged_df)
        st.subheader("Regression Results")
        st.write(model.summary())

        # ---- Forecast UI ----
        st.subheader("Forecast Direct Sales")
        min_meta, max_meta = (
            float(merged_df["Meta Spend"].min()),
            float(merged_df["Meta Spend"].max()),
        )
        min_google, max_google = (
            float(merged_df["Google Spend"].min()),
            float(merged_df["Google Spend"].max()),
        )
        meta_input = st.slider(
            "Meta Spend", min_meta, max_meta, (min_meta + max_meta) / 2
        )
        google_input = st.slider(
            "Google Spend", min_google, max_google, (min_google + max_google) / 2
        )
        if st.button("Forecast Sales"):
            prediction = forecast_fn(meta_input, google_input)
            st.metric(label="Predicted Direct Sales", value=f"${prediction:,.2f}")

        # ---- Visualization ----
        st.subheader("Historical vs Fitted Direct Sales")
        X = merged_df[["Meta Spend", "Google Spend"]]
        X = sm.add_constant(X)
        merged_df["Fitted"] = model.predict(X)
        plot_df = merged_df.set_index("Month Start")[["Direct Sales", "Fitted"]]
        st.line_chart(plot_df)
