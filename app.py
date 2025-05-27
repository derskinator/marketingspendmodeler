import streamlit as st
import pandas as pd
from regression_model import clean_shopify_data, merge_spend_data, run_regression

st.set_page_config(page_title="Spend vs Sales Regression", layout="wide")

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
        # Load and concatenate Shopify data
        shop_dfs = [pd.read_csv(f) for f in shop_files]
        shop_df = pd.concat(shop_dfs, ignore_index=True)

        # Parse Month Start from filenames or columns
        # Expecting a 'Month Start' column or parse from filename if needed
        # -- user to supply Month Start in combined df

        # Clean Shopify
        direct_sales_df = clean_shopify_data(shop_df)

        # Load and clean spend files
        meta_df = pd.read_csv(meta_file)
        google_df = pd.read_csv(google_file)

        # User should preprocess meta_df and google_df similarly to examples
        # For now, assume they have 'Month Start', 'Meta Spend', 'Google Spend'

        # Merge data
        merged_df = merge_spend_data(direct_sales_df, meta_df, google_df)
        st.subheader("Merged Sales & Spend Data")
        st.dataframe(merged_df)

        # Run regression
        model, forecast_fn = run_regression(merged_df)

        # Display regression summary
        st.subheader("Regression Results")
        st.write(model.summary())

        # Forecast inputs
        st.subheader("Forecast Direct Sales")
        meta_input = st.slider("Meta Spend", float(merged_df['Meta Spend'].min()), float(merged_df['Meta Spend'].max()), float(merged_df['Meta Spend'].mean()))
        google_input = st.slider("Google Spend", float(merged_df['Google Spend'].min()), float(merged_df['Google Spend'].max()), float(merged_df['Google Spend'].mean()))
        if st.button("Forecast Sales"):
            prediction = forecast_fn(meta_input, google_input)
            st.metric(label="Predicted Direct Sales", value=f"${prediction:,.2f}")

        # Visualization
        st.subheader("Historical vs Fitted Direct Sales")
        merged_df['Fitted'] = model.predict(
            sm.add_constant(merged_df[['Meta Spend', 'Google Spend']])
        )
        st.line_chart(merged_df.set_index('Month Start')[['Direct Sales', 'Fitted']])
