import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score

from toy_graph.model import DeepWalk
from toy_graph.util import read_content_file

g = nx.read_edgelist('../data/cora.cites', delimiter='\t', create_using=nx.Graph, nodetype=str)
model = DeepWalk()
model.fit(g, n_jobs=4)

# visualize node embedding
node_embedding = model.get_node_embedding()
nodes, labels, _ = read_content_file('../data/cora.content')
embeddings = [node_embedding[n] for n in nodes]

trans = TSNE(n_components=2)
X_reduced = trans.fit_transform(embeddings)

df = pd.DataFrame()
df['comp-1'] = X_reduced[:, 0]
df['comp-2'] = X_reduced[:, 1]
df['y'] = labels

sns.scatterplot(x="comp-1",
                y="comp-2",
                hue=df.y.tolist(),
                palette=sns.color_palette("hls", len(pd.unique(df['y']))),
                data=df).set(title="embedding T-SNE projection")
plt.savefig("node2vec_cora.png")

# node classification
X, y = embeddings, labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
clf = LogisticRegressionCV(
    Cs=10,
    cv=10,
    scoring="accuracy",
    verbose=1,
    multi_class="ovr",
    max_iter=300
)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('acc: ', acc)
