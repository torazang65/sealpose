import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
