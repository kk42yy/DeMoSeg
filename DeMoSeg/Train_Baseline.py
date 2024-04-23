import os
from training.trainer.BaseTrainer import BasicTrainer

if __name__ == "__main__":
    basic_trainer = BasicTrainer(
        task='2020', 
        fold=0,
        basepath=os.path.split(os.path.abspath(__file__))[0]
    )
    basic_trainer.initialize()
    basic_trainer.run_train()
