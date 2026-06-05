from toy_graph.model.embedder import RandomWalkEmbedder


class Node2vec(RandomWalkEmbedder):
    """node2vec with default return/in-out parameters."""

    def __init__(
        self,
        p: float = 0.5,
        q: float = 2.0,
        num_walks: int = 10,
        walk_length: int = 20,
        epochs: int = 10,
        window_size: int = 5,
        embed_size: int = 128,
        weight_attr_name=None,
        random_state=None,
    ):
        super().__init__(
            p=p,
            q=q,
            num_walks=num_walks,
            walk_length=walk_length,
            epochs=epochs,
            window_size=window_size,
            embed_size=embed_size,
            weight_attr_name=weight_attr_name,
            random_state=random_state,
        )
