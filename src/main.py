import os
import streamlit as st
import pandas as pd
from gensim.models import FastText
import plotly.express as px
from sklearn.decomposition import PCA

MODELS_DIR = "./models"

st.set_page_config(page_title="FastText Viewer", layout="wide")

def get_model_files(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return [f for f in os.listdir(directory) if f.endswith('.model')]

@st.cache_resource
def load_fasttext_model(filepath):
    return FastText.load(filepath)

model_files = get_model_files(MODELS_DIR)

if model_files:
    selected_model_file = st.selectbox("Select model:", model_files)
    model_path = os.path.join(MODELS_DIR, selected_model_file)
    try:
        model = load_fasttext_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
        st.warning(f"No `.model` files found in directory: `{os.path.abspath(MODELS_DIR)}`")

if model:
    vocab_size = len(model.wv)
    vector_dim = model.wv.vector_size
else:
    vocab_size, vector_dim = 0, 0

st.title("FastText viewer — Hexaemeron lemmatized corpus")
st.caption(f"{vocab_size} vocab · {vocab_size} vectors shipped · dim={vector_dim} · trained in FastText")

if "query_word" not in st.session_state:
    st.session_state["query_word"] = "μελισσα"

col1, col2 = st.columns(2)

with col1:
    st.subheader("NEAREST NEIGHBOURS")

    input_col, btn_col = st.columns([3, 1])
    with input_col:
        current_input = st.text_input("Search word", value=st.session_state["query_word"], label_visibility="collapsed")
    with btn_col:
        search_clicked = st.button("Search", use_container_width=True)

    st.caption("Cosine similarity over the trained word vectors. Click any neighbour to query it.")

    if search_clicked or current_input != st.session_state["query_word"]:
        st.session_state["query_word"] = current_input

    if st.session_state["query_word"] and model:
        try:
            similar_words = model.wv.most_similar(st.session_state["query_word"], topn=15)
            df_neighbours = pd.DataFrame(similar_words, columns=["Word", "Score"])
            df_neighbours["Score"] = df_neighbours["Score"].round(3)

            event = st.dataframe(
                df_neighbours,
                use_container_width=True,
                hide_index=True,
                selection_mode="single-row",
                on_select="rerun"
            )

            if event and event.selection["rows"]:
                selected_row = event.selection["rows"][0]
                clicked_word = df_neighbours.iloc[selected_row]["Word"]
                if clicked_word != st.session_state["query_word"]:
                    st.session_state["query_word"] = clicked_word
                    st.rerun()

        except KeyError:
            st.error(f"Word '{st.session_state['query_word']}' not found in model vocabulary.")
        except Exception as e:
            st.error(f"Search error: {e}")

with col2:
    st.subheader("VOCABULARY BROWSER")

    filter_query = st.text_input("Filter vocabulary", placeholder="Filter the top vocabulary...", label_visibility="collapsed")
    st.caption("Click a row to look up its neighbours and add it to the plot. Click column headers to sort.")

    if model:
        vocab_list = []
        for word in model.wv.key_to_index.keys():
            freq = model.wv.get_vecattr(word, "count") if model.wv.has_index_for(word) else None
            vocab_list.append({"Word": word, "Frequency": freq})

        df_vocab = pd.DataFrame(vocab_list)
        if filter_query:
            df_vocab = df_vocab[df_vocab["Word"].str.contains(filter_query, case=False, na=False)]

        st.dataframe(
            df_vocab,
            use_container_width=True,
            hide_index=True
        )

st.divider()
st.subheader("2-D PCA PROJECTION")

if st.session_state["query_word"] and model:
    try:
        query = st.session_state["query_word"]

        sim_words = model.wv.most_similar(query, topn=15)
        words_to_plot = [query] + [w for w, _ in sim_words]

        vectors = [model.wv[w] for w in words_to_plot]

        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(vectors)

        df_pca = pd.DataFrame(pca_coords, columns=["PC1", "PC2"])
        df_pca["Word"] = words_to_plot
        df_pca["Role"] = ["Query word"] + ["Nearest neighbor"] * len(sim_words)

        fig = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            text="Word",
            color="Role",
            color_discrete_map={"Query word": "#d9534f", "Nearest neighbor": "#0275d8"},
            hover_name="Word"
        )
        fig.update_traces(textposition='top center', marker=dict(size=10))
        fig.update_layout(height=500, xaxis_title="Principal Component 1", yaxis_title="Principal Component 2")

        st.plotly_chart(fig, use_container_width=True)

    except Exception:
        st.info("Unable to generate PCA plot for the current query.")
