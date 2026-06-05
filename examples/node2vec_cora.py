from toy_graph import Node2vec

from cora_pipeline import run_cora_demo

if __name__ == "__main__":
    run_cora_demo(Node2vec(random_state=42), output_image="node2vec_cora.png")
