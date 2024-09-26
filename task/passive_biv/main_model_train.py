from task.passive_biv.fe_heart_sage_v1.train.model import FEHeartSAGETrainer
from task.passive_biv.fe_heart_sage_v2.train.model import FEHeartSageV2Trainer

if __name__ == "__main__":
    model = FEHeartSAGETrainer()
    model.train()
