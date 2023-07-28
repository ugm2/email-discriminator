import io
import os

import pandas as pd
import streamlit as st

from email_discriminator.core.data_versioning import GCSVersionedDataHandler

# -- Set page config
apptitle = "TLDR"

st.set_page_config(page_title=apptitle, page_icon="📧")

# Title the app
st.title("📧 TLDR email discriminator")

# Initialize GCS handler
BUCKET_NAME = os.getenv("BUCKET_NAME", "email-discriminator")
gcs_handler = GCSVersionedDataHandler(BUCKET_NAME)


@st.cache_data
def load_data(selected_csv: str) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(selected_csv))


@st.cache_data
def download_data() -> pd.DataFrame:
    return gcs_handler.download_new_predicted_data()


def main():
    st.sidebar.title("Predicted Data in GCS")

    # Fetch CSV files from GCS
    predict_csv_files = download_data()

    # Create a dropdown in the sidebar to select a CSV file
    selected_csv = st.sidebar.selectbox("Select a CSV file", predict_csv_files.keys())

    # Load the CSV file
    df = load_data(predict_csv_files[selected_csv])

    # Checkbox to control whether to show all data or just relevant data
    show_all = st.sidebar.checkbox("Show all data")

    if not show_all:
        df = df[df["predicted_is_relevant"] == True]

    # Pagination
    page_size = 5
    page_num = (
        st.number_input(
            label="Page Number", min_value=1, max_value=len(df) // page_size + 1, step=1
        )
        - 1
    )

    # Get the relevant rows
    start = page_num * page_size
    end = (page_num + 1) * page_size
    relevant_df = df.iloc[start:end, :]

    # Loop through each row in the dataframe
    for _, row in relevant_df.iterrows():
        # Try to split the content, but allow for exceptions if the format is unexpected
        try:
            # Parse the article into title, link, and summary
            title, link_summary = row["article"].split("\n", 1)
            link, summary = link_summary.rsplit("\n", 1)
            link = link.strip("[]")
        except ValueError:
            title = row["article"]
            link = ""
            summary = ""

        # Use markdown to present the article
        st.markdown(f"## {title}")
        if link:
            st.markdown(f"[Read more]({link})")
        if summary:
            st.markdown(summary)
        st.markdown("---")


if __name__ == "__main__":
    main()
