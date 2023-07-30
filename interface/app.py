import io

import data_handler as dh
import pandas as pd
import streamlit as st

# -- Set page config
apptitle = "TLDR"
st.set_page_config(page_title=apptitle, page_icon="📧")

# Title the app
st.title("📧 TLDR email discriminator")


def load_data(selected_csv: str) -> pd.DataFrame:
    """
    Loads data from the specified CSV file.
    """
    return pd.read_csv(io.StringIO(selected_csv))


def get_relevant_df(df: pd.DataFrame, show_all: bool) -> pd.DataFrame:
    """
    Returns a dataframe containing only the relevant rows, unless show_all is True.
    """
    if not show_all:
        relevant_df = df[(df["predicted_is_relevant"] == True)]
    else:
        relevant_df = df
    return relevant_df


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
    # Fetch CSV files from GCS
    predict_csv_files = dh.download_data()

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
        # Upload the data
        dh.upload_reviewed_data(reviewed_df, unreviewed_df, selected_csv)
        st.sidebar.success("Data uploaded to GCS 😊")


if __name__ == "__main__":
    main()
