from toy_graph import DeepWalk

from cora_pipeline import run_cora_demo

if __name__ == "__main__":
    run_cora_demo(DeepWalk(random_state=42), output_image="deep_walk_cora.png")
