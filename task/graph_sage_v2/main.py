from task.graph_sage_v2.train.model import GraphSAGETrainer

if __name__ == "__main__":
    # fetch config path and serve to our model
    model = GraphSAGETrainer()
    model.train()
