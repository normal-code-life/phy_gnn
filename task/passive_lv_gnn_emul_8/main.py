from task.passive_lv_gnn_emul_8.train.model import PassiveLvGNNEmulTrainer

if __name__ == "__main__":
    # fetch config path and serve to our model
    model = PassiveLvGNNEmulTrainer()
    model.train()
