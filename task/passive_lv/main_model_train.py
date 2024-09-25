# from task.passive_lv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer
from task.passive_lv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer

if __name__ == "__main__":
    # fetch config path and serve to our model
    model = FEHeartSageV2Trainer()
    model.train()
