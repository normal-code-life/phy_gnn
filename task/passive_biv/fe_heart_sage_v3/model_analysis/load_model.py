
import torch


stats_dict = torch.load("ckpt_34900.pth", map_location=torch.device('cpu'), weights_only=False)
for i in stats_dict["model_state_dict"]:
    value = stats_dict["model_state_dict"][i]
    print(i, value.shape, torch.max(value), torch.min(value))

print("===========================================================================")

stats_dict2 = torch.load("ckpt.pth", map_location=torch.device('cpu'), weights_only=False)
for i in stats_dict["model_state_dict"]:
    value = stats_dict2["model_state_dict"][i]
    print(i, value.shape, torch.max(value), torch.min(value))