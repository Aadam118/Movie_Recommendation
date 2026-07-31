import pickle
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ----------------------------------------------------------------------------
# CACHED DATA LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_data():
    """Load the precomputed movies dataframe and similarity matrix."""
    movies = pickle.load(open("df.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))

    # df.pkl may be a dict (from df.to_dict()) — normalize to DataFrame
    if isinstance(movies, dict):
        movies = pd.DataFrame(movies)

    return movies, similarity


def recommend(movie_title: str, movies: pd.DataFrame, similarity):
    """Return the top 5 recommended movie titles for a given movie."""
    try:
        movie_index = movies[movies["title"] == movie_title].index[0]
    except IndexError:
        return []

    distances = similarity[movie_index]
    movie_list = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i]["title"] for i, _score in movie_list]


# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("🎬 CineMatch")
st.write("Discover your next favorite movie — powered by Machine Learning.")

# ----------------------------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------------------------
try:
    movies_df, similarity_matrix = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False
    st.error(
        "⚠️ Could not find `df.pkl` / `similarity.pkl`. "
        "Please make sure both files are in the app's root directory."
    )

# ----------------------------------------------------------------------------
# MAIN UI
# ----------------------------------------------------------------------------
if data_loaded:
    selected_movie = st.selectbox(
        "Search and select a movie you like:",
        options=movies_df["title"].values,
        index=None,
        placeholder="e.g. The Dark Knight",
    )

    if st.button("Recommend"):
        if not selected_movie:
            st.warning("Please select a movie first.")
        else:
            recommendations = recommend(selected_movie, movies_df, similarity_matrix)

            if not recommendations:
                st.error("Sorry, no recommendations found for that movie.")
            else:
                st.markdown(f"### Because you liked *{selected_movie}*")
                for i, title in enumerate(recommendations, start=1):
                    st.write(f"{i}. {title}")
