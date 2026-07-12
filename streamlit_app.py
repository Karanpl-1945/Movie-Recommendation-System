import streamlit as st

from models.recommend import movie_recommend


st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #111827 45%, #1f2937 100%);
        color: #f8fafc;
    }
    .hero-card {
        padding: 1.5rem 1.75rem;
        border-radius: 1.25rem;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.25);
    }
    .subtle {
        color: #cbd5e1;
        font-size: 0.98rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <h1 style="margin-bottom: 0.35rem;">Movie Recommendation System</h1>
        <div class="subtle">Hybrid recommendations from content similarity, item-based CF, and popularity.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("recommendation_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        movie_title = st.text_input("Movie title", value="Toy Story", placeholder="Enter a movie title")
    with col2:
        no_recommendation = st.slider("Recommendations", min_value=1, max_value=19, value=5)

    submitted = st.form_submit_button("Get recommendations")


if submitted:
    if not movie_title.strip():
        st.warning("Enter a movie title first.")
    else:
        try:
            result = movie_recommend(movie_title, no_recommendation)

            if result.get("message") == "Movie not found":
                st.error(f'No model match found for "{movie_title}".')
            else:
                st.success(f'Recommendations for "{result["query_movie"]}"')
                recommendations = result.get("recommendations", [])

                if not recommendations:
                    st.info("No recommendations available for this title.")
                else:
                    for item in recommendations:
                        with st.container(border=True):
                            left, right = st.columns([4, 1])
                            with left:
                                st.subheader(item["title"])
                                st.caption(f'Movie ID: {item["movieId"]}')
                                st.write(
                                    f'Average rating: {item["average_rating"]:.2f} | '
                                    f'Rating count: {item["rating_count"]}'
                                )
                            with right:
                                st.metric(
                                    "Confidence",
                                    f'{item.get("confidence", 0.0):.3f}',
                                )
                                st.progress(min(max(float(item.get("confidence", 0.0)), 0.0), 1.0))
        except Exception as exc:
            st.error(f"Recommendation failed: {exc}")

st.markdown(
    "---\n"
    "Use the FastAPI docs at `/docs` for the API version, or this page for a quick UI demo.",
    unsafe_allow_html=False,
)