import multiprocessing

from task.passive_biv.train.model import PassiveBiVTrainer

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    # fetch config path and serve to our model
    model = PassiveBiVTrainer()
    model.train()
