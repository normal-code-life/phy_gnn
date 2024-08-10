import multiprocessing

from task.passive_lv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # fetch config path and serve to our model
    model = FEHeartSAGETrainer()
    model.train()
