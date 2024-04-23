import os
from training.trainer.DeMoSeg_Trainer import DeMoSeg_Trainer

if __name__ == "__main__":
    demoseg_trainer = DeMoSeg_Trainer(
        task='2020', 
        fold=0,
        basepath=os.path.split(os.path.abspath(__file__))[0]
    )
    demoseg_trainer.initialize()
    demoseg_trainer.run_train()
