from task.passive_lv.fe_heart_sage_v1.train.model import GraphSAGETrainer

if __name__ == "__main__":
    # fetch config path and serve to our model
    model = GraphSAGETrainer()
    model.train()
