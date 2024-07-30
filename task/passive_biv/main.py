import multiprocessing

from task.passive_biv_v1.train.model import GraphSAGETrainer

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # fetch config path and serve to our model
    model = GraphSAGETrainer()
    model.train()
