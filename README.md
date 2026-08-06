# Embedding Viewer

A Streamlit-based web application for exploring and visualizing word embeddings. This tool supports various model formats including FastText, Word2Vec, and GloVe, allowing users to inspect vocabulary, find nearest neighbors, and visualize semantic relationships via PCA.

## Features

- Model Loading: Supports .model, .vec, .txt, .bin, .kv, and .glove formats.
- Nearest Neighbors: Find and navigate through semantically similar words using cosine similarity.
- Vocabulary Browser: Filter and sort through the model's entire vocabulary, including frequency counts where available.
- 2D Visualization: Automatic PCA (Principal Component Analysis) projection of the query word and its neighbors for spatial intuition.
- Interactive UI: Click-to-search functionality within the neighbors list and vocabulary table.

## Installation

1. Clone the repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your pre-trained model files in the `./models` directory.
2. Run the Streamlit application:
   ```bash
   streamlit run src/main.py
   ```
3. Use the dropdown menu to select the model you wish to explore.

## Requirements

- Python 3.x
- gensim
- streamlit
- pandas
- scikit-learn
- plotly
