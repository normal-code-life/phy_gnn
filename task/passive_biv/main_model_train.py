# import multiprocessing

from task.passive_biv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")

    model = FEHeartSageV2Trainer()
    model.train()
