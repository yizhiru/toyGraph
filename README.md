# toyGraph
Toy Graph Learning package, inspired by StellarGraph.

---------
实现 Graph Learning 算法：
1. DeepWalk
2. node2vec

## 安装

```bash
pip install -e .
# 运行 examples 还需可选依赖
pip install -e ".[examples]"
```

## 使用

```python
from toy_graph import DeepWalk, Node2vec
import networkx as nx

g = nx.karate_club_graph()
model = DeepWalk(random_state=42)
model.fit(g, n_jobs=4)
embeddings = model.get_node_embedding()
```

完整 Cora 示例见 `examples/`。

