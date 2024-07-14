from task.graph_sage_v2.train.model import GraphSAGETrainer
import multiprocessing


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    # fetch config path and serve to our model
    model = GraphSAGETrainer()
    model.train()
