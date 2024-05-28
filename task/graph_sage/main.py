from task.graph_sage.train.model import GraphSAGETrainer

if __name__ == "__main__":
    # fetch config path and serve to our model
    model = GraphSAGETrainer()
    model.train()
