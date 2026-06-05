"""Shared Cora demo pipeline for DeepWalk and node2vec."""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegressionCV
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from toy_graph.model.embedder import RandomWalkEmbedder
from toy_graph.util import read_content_file

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def run_cora_demo(model: RandomWalkEmbedder, output_image: str, n_jobs: int = 4) -> float:
    graph = nx.read_edgelist(
        DATA_DIR / "cora.cites",
        delimiter="\t",
        create_using=nx.Graph,
        nodetype=str,
    )
    model.fit(graph, n_jobs=n_jobs)

    node_embedding = model.get_node_embedding()
    nodes, labels, _ = read_content_file(DATA_DIR / "cora.content")
    embeddings = [node_embedding[n] for n in nodes]

    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(embeddings)

    df = pd.DataFrame({
        "comp-1": reduced[:, 0],
        "comp-2": reduced[:, 1],
        "y": labels,
    })
    sns.scatterplot(
        x="comp-1",
        y="comp-2",
        hue="y",
        palette=sns.color_palette("hls", len(pd.unique(df["y"]))),
        data=df,
    ).set(title="embedding T-SNE projection")
    plt.savefig(output_image)
    plt.close()

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, test_size=0.1, random_state=42,
    )
    clf = LogisticRegressionCV(
        Cs=10,
        cv=10,
        scoring="accuracy",
        multi_class="ovr",
        max_iter=300,
    )
    clf.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, clf.predict(X_test))
    print(f"accuracy: {accuracy:.4f}")
    return accuracy
