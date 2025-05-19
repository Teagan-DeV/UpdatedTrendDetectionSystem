import streamlit as st
import matplotlib.pyplot as plt
from trend_backend import generate_trends

# ========================
# --- PAGE CONFIGURATION
# ========================
st.set_page_config(page_title="Trend Tracker", layout="wide")

st.sidebar.title("Settings")
test_mode = st.sidebar.toggle("Test Mode", value=True)

st.write(f"✅ Test mode is {'ON' if test_mode else 'OFF'}")

# Set limits based on toggle
per_language_limit = 2 if test_mode else 40
max_pages = 1 if test_mode else 5

# ========================
# --- LANDING PAGE
# ========================
if "view" not in st.session_state:
    st.session_state.view = None

st.title("🌿 Trend Tracker")
st.markdown("### What would you like to see today?")

col1, col2 = st.columns(2)

with col1:
    if st.button("🟢 Generate This Week’s Trends"):
        st.session_state.view = "weekly"

with col2:
    if st.button("🔵 Generate This Month’s Trends"):
        st.session_state.view = "monthly"

# ========================
# --- RESULTS VIEW
# ========================
if st.session_state.view:
    mode = st.session_state.view
    st.markdown("---")
    st.subheader(f"📈 {mode.title()} Trends")
    
    cluster_labels, final_articles, wordcloud_figs, trend_fig = generate_trends(mode, test_mode)

    for i in range(len(cluster_labels)):
        st.markdown(f"#### 🧠 Cluster {i+1}: {cluster_labels[i]['custom_label']}")

        col1, col2, col3 = st.columns([1.2, 1, 1])

        with col1:
            st.markdown("**Top Terms**")
            for term in cluster_labels[i]["top_terms"]:
                st.markdown(f"- {term}")

        with col2:
            st.markdown("**Sample Articles**")
            top_terms = cluster_labels[i]["top_terms"]
            shown_terms = set()

            for term in top_terms:
                match = next(
                    (a for a in final_articles 
                     if a["cluster"] == i and term in a.get("title", "").lower()),
                     None
                )
                if match:
                    shown_terms.add(term)
                    headline = match.get("title", "Untitled")
                    snippet = match.get("description") or match.get("content) or "No summary available"
                    url = match.get("url", "")
        
                    st.markdown(f"- *{headline}*")
                    st.markdown(f"{snippet}")
                    if url:
                        st.markdown(f"[Read Full Article]({url})", unsafe_allow_html=True)
                    st.markdown("---")

            if not shown_terms:
                st.markdown("*No article headlines matched the top terms*")


        with col3:
            st.markdown("**Word Cloud**")
            st.pyplot(wordcloud_figs[i])

    st.markdown("---")
    st.subheader("📊 Overall Trend Chart")
    st.pyplot(trend_fig)

# ========================
# --- FOOTER
# ========================
st.markdown("<hr><p style='text-align:center;'>Powered by automated keyword analysis and article clustering.</p>", unsafe_allow_html=True)
