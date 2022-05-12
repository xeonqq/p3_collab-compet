import numpy as np
import torch
from device import device


def separate_and_group_experiences(experiences):
    sampled_items = ([],[],[],[],[])
    for experience in experiences:
        for i,items in enumerate(experience):
            sampled_items[i].append(items)

    torch_sampled_items = []
    for items in sampled_items:
        torch_items = torch.from_numpy(np.vstack(items)).float().to(device)
        torch_sampled_items.append(torch_items)

    return torch_sampled_items

