import io
import time

import pandas as pd
import streamlit as st

import interface.data_handler as dh
from interface.prefect_client import call_flow

# Set page config
apptitle = "TLDR"
st.set_page_config(page_title=apptitle, page_icon="📧")

# Title the app
st.sidebar.title("📧 TLDR email discriminator")


@st.cache_data
def download_data():
    return dh.download_data()


def refresh_data():
    st.cache_data.clear()
    del st.session_state.df


def load_data(selected_csv: str) -> pd.DataFrame:
    """
    Loads data from the specified CSV file.
    """
    return pd.read_csv(io.StringIO(selected_csv))


def get_relevant_df(df: pd.DataFrame, show_all: bool) -> pd.DataFrame:
    """
    Returns a dataframe containing only the relevant rows, unless show_all is True.
    """
    if show_all:
        return df
    else:
        return df[df["predicted_is_relevant"].astype(bool)]


def update_df(index: int, predicted_is_relevant: bool, row: pd.Series) -> bool:
    """
    Updates the 'predicted_is_relevant' field of the specified row in the dataframe.
    Returns True if the field was changed, and False otherwise.
    """
    if predicted_is_relevant != row["predicted_is_relevant"]:
        st.session_state.df.loc[index, "predicted_is_relevant"] = int(
            predicted_is_relevant
        )
        st.experimental_rerun()
    return False


def main():
    # Refresh cache
    if st.sidebar.button("Fetch new data & clear cache"):
        refresh_data()

    # Fetch CSV files from GCS
    predict_csv_files = download_data()

    # Create a dropdown in the sidebar to select a CSV file
    selected_csv = st.sidebar.selectbox("Select a CSV file", predict_csv_files.keys())

    # Load the CSV file
    if selected_csv is None:
        st.warning("No data in the server 🥲")
        return
    df = load_data(predict_csv_files[selected_csv])

    # If dataframe is not in session_state, add it
    if "df" not in st.session_state:
        st.session_state.df = df

    # Checkbox to control whether to show all data or just relevant data
    show_all = st.sidebar.checkbox("Show all data")
    relevant_df = get_relevant_df(st.session_state.df, show_all)

    # Pagination
    page_size = 5
    page_num = (
        st.sidebar.number_input(
            label="Page Number",
            min_value=1,
            max_value=len(st.session_state.df) // page_size + 1,
            step=1,
        )
        - 1
    )

    # Get the relevant rows
    start_pagination = page_num * page_size
    end_pagination = (page_num + 1) * page_size
    relevant_df = relevant_df.iloc[start_pagination:end_pagination]

    # Loop through each row in the dataframe
    for index, row in relevant_df.iterrows():
        # Checkbox for selecting article
        predicted_is_relevant = st.checkbox(
            f"Article {index} is relevant", value=row["predicted_is_relevant"]
        )
        is_relevant_changed = update_df(index, predicted_is_relevant, row)

        if not is_relevant_changed or (is_relevant_changed and show_all):
            # Parse the article into title, link, and summary
            title, link_summary = row["article"].split("\n", 1)
            link, summary = link_summary.rsplit("\n", 1)
            link = link.strip("[]")

            # Use markdown to present the article
            st.markdown(f"## {title}")
            if link:
                st.markdown(f"[Read more]({link})")
            if summary:
                st.markdown(summary)
            st.markdown("---")

    if st.sidebar.button("Upload Reviewed Data"):
        # Split the dataframe into "reviewed" and "unreviewed" datasets
        reviewed_df, unreviewed_df = dh.split_dataframe(
            st.session_state.df, end_pagination
        )
        # Add column to reviewed_df 'is_relevant' equal to 'predicted_is_relevant'
        reviewed_df["is_relevant"] = reviewed_df["predicted_is_relevant"]
        # Upload the data
        dh.upload_reviewed_data(reviewed_df, unreviewed_df, selected_csv)
        # Refresh data
        refresh_data()
        # Refresh page
        st.sidebar.success("Data uploaded to GCS 📚😊")
        time.sleep(2)
        st.experimental_rerun()

    st.sidebar.write("---")
    st.sidebar.write("### Training")
    with st.sidebar.expander("Training Config", False):
        deployment_name = st.sidebar.text_input(
            "Flow deployment name", "train-flow/email_discriminator-train"
        )
        model_stage = st.sidebar.selectbox(
            "Model stage", ["Staging", "Production"], index=0
        )
    if st.sidebar.button("Train"):
        # Call the training flow
        flow_run = call_flow(
            deployment_name=deployment_name, parameters={"model_stage": model_stage}
        )
        st.sidebar.success(
            f"Training flow started with Flow Run ID **{flow_run.id}** and Flow Run name **{flow_run.name}** 🚗💨"
        )


if __name__ == "__main__":
    main()
