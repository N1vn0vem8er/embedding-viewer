import os
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from embeddingsutils import (
    get_model_files,
    load_embedding_model,
    get_wv,
    get_similar_words
)

MODELS_DIR = "./models"

st.set_page_config(page_title="Embedding Viewer", layout="wide")

model_files = get_model_files(MODELS_DIR)

if model_files:
    selected_model_file = st.selectbox("Select model:", model_files)
    model_path = os.path.join(MODELS_DIR, selected_model_file)
    try:
        model = load_embedding_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model = None
else:
    st.warning(f"No valid model files found in directory: `{os.path.abspath(MODELS_DIR)}`")

wv = get_wv(model) if model else None

if wv:
    vocab_size = len(wv)
    vector_dim = wv.vector_size
    model_type = type(model).__name__
else:
    vocab_size, vector_dim = 0, 0
    model_type = ""

st.title("Embedding viewer")
st.caption(f"{vocab_size} vocab · {vocab_size} vectors shipped · dim={vector_dim} · trained in {model_type}")

tab1, tab2, tab3 = st.tabs([
    "Exploration",
    "Out Of Vocabulary",
    "Clustering"
])

with tab1:

    if "query_word" not in st.session_state:
        st.session_state["query_word"] = "μελισσα"

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Nearest Neighbours")

        metric_options = {
                    "Cosine similarity": "cosine",
                    "Euclidean distance": "euclidean",
                    "Manhattan distance": "manhattan",
                    "Chebyshev distance": "chebyshev",
                    "Dot product": "dot"
                }
        selected_metric_label = st.selectbox("Distance Metric", list(metric_options.keys()))
        selected_metric = metric_options[selected_metric_label]

        normalize_vectors = st.checkbox(
                    "Normalize vectors",
                    value=False,
                    help="Scales the length of each vector to 1.0. Enable this option to even out the effects of word frequency."
                )

        input_col, btn_col = st.columns([3, 1])
        with input_col:
            current_input = st.text_input("Search word", value=st.session_state["query_word"], label_visibility="collapsed")
        with btn_col:
            search_clicked = st.button("Search", use_container_width=True)

        st.caption(f"Finding neighbours using {selected_metric_label.lower()}. Click any neighbour to query it.")

        if search_clicked or current_input != st.session_state["query_word"]:
            st.session_state["query_word"] = current_input

        if st.session_state["query_word"] and wv:
            try:
                similar_words = get_similar_words(wv, st.session_state["query_word"], metric=selected_metric, topn=15, normalize_vectors=normalize_vectors)
                val_col_name = "Score" if selected_metric in ["cosine", "dot"] else "Distance"
                df_neighbours = pd.DataFrame(similar_words, columns=["Word", val_col_name])
                df_neighbours[val_col_name] = df_neighbours[val_col_name].round(4)

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
        st.subheader("Vocabulary Browser")

        filter_query = st.text_input("Filter vocabulary", placeholder="Filter the top vocabulary...", label_visibility="collapsed")
        st.caption("Click a row to look up its neighbours and add it to the plot. Click column headers to sort.")

        if wv:
            vocab_list = []
            for word in wv.key_to_index.keys():
                try:
                    freq = wv.get_vecattr(word, "count")
                except (ValueError, KeyError, AttributeError):
                    freq = None

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
    st.subheader("2-D PCA Projection")

    if st.session_state["query_word"] and wv:
        try:
            query = st.session_state["query_word"]
            sim_words = get_similar_words(wv, query, metric=selected_metric, topn=15, normalize_vectors=normalize_vectors)
            words_to_plot = [query] + [w for w, _ in sim_words]

            vectors = [wv[w] for w in words_to_plot]

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
with tab2:
    st.subheader("Out Of Vocabulary Resilience")

    oov_word = st.text_input("Enter a made-up or misspelled word:")

    if oov_word and wv:
        in_vocab = oov_word in wv.key_to_index

        if in_vocab:
            st.success(f"Word '{oov_word}' is in the vocabulary.")
        else:
            st.warning(f"Word '{oov_word}' is not in the vocabulary.")
            try:
                vec = wv[oov_word]
                st.success(f"FastText successfully generated an n-gram embedding for '{oov_word}'.")

                st.write("Its closest neighbors are:")
                sims = wv.most_similar(positive=[vec], topn=10)
                st.dataframe(pd.DataFrame(sims, columns=["Neighbor in Vocab", "Similarity Score"]))
            except KeyError:
                st.error("This model does not support subword embeddings (it might be a standard Word2Vec/GloVe or missing n-gram data).")

with tab3:
    st.subheader("Word / Concept Clustering (K-Means)")
    st.markdown("Enter a list of words (separated by commas). The model will convert them into vectors, and K-Means will divide them into groups of similar meaning.")

    words_input = st.text_area("List of words:", value="Επιστήμη, Θεωρία, Μαθηματικά, Φιλοσοφία, Λογική, Αγάπη, Θυμός, Χαρά, Φόβος, Λύπη")
    k_clusters = st.slider("Number of clusters (K):", min_value=2, max_value=10, value=4)

    if st.button("Cluster Words") and wv:
        words = [w.strip() for w in words_input.split(",") if w.strip()]
        valid_words = []
        vectors = []

        for w in words:
            try:
                vectors.append(wv[w])
                valid_words.append(w)
            except KeyError:
                pass

        if len(valid_words) >= k_clusters:
            kmeans = KMeans(n_clusters=k_clusters, random_state=42)
            labels = kmeans.fit_predict(normalize(vectors))

            pca = PCA(n_components=2)
            coords = pca.fit_transform(normalize(vectors))

            df_cluster = pd.DataFrame({
                "Word": valid_words,
                "Cluster": [f"Cluster {l}" for l in labels],
                "PC1": coords[:, 0],
                "PC2": coords[:, 1]
            })

            fig2 = px.scatter(
                df_cluster, x="PC1", y="PC2", color="Cluster", text="Word", size_max=60
            )
            fig2.update_traces(textposition='top right', marker=dict(size=12))
            st.plotly_chart(fig2, use_container_width=True)

            st.dataframe(df_cluster[["Cluster", "Word"]].sort_values("Cluster"), hide_index=True)
        else:
            st.error(f"Not enough valid words found in the model to form {k_clusters} clusters.")
