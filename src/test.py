import numpy as np
from wandb_wrapper import WandbWrapper


wdb = WandbWrapper("../config/test.yaml")

for i in range(10):
    wdb.log(
        {"_step": i, "something": np.random.randint(10), "anything": np.random.random()}
    )

wdb.finish()
