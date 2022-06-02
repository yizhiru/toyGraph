from toy_graph.model import Node2vec


class DeepWalk(Node2vec):
    """
    Deep walk算法实现
    """

    def __init__(self,
                 num_walks=10,
                 walk_length=20,
                 epochs=10,
                 window_size=5,
                 embed_size=128,
                 weight_attr_name=None):
        super(DeepWalk, self).__init__(1.0,
                                       1.0,
                                       num_walks,
                                       walk_length,
                                       epochs,
                                       window_size,
                                       embed_size,
                                       weight_attr_name)
