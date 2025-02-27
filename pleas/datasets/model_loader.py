import torch
from torchvision import models



def load_model(name, variant=None):
    if name == "rn50":
        if variant is None:
            model = models.resnet50()
        elif variant == "v1a":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif variant == "v2a":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif variant == "v1b":
            model = models.resnet50()
            model.load_state_dict({
                k[7:]: v
                for k, v in torch.load("/path/to/model")[
                    "state_dict"
                ].items()
            })
        elif variant == "v2b":
            model = models.resnet50()
            model.load_state_dict({
                k[7:]: v
                for k, v in torch.load(
                    "/path/to/model"
                )['model_ema'].items()
                if "module" in k
            })
        else:
            raise ValueError(f"Unknown variant {variant}")

    elif name == "rn18":
        if variant is None:
            model = models.resnet18()
        elif variant == "v1a":
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif variant == "v1b":
            model = models.resnet18()
            model.load_state_dict({
                k[7:]: v
                for k, v in torch.load("/path/to/model")[
                    "state_dict"
                ].items()
            })
        else:
            raise ValueError(f"Unknown variant {variant}")

    elif name == "rn101":
        if variant is None:
            model = models.resnet101()
        elif variant == "v2a":
            model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
        elif variant == "v2b":
            model = models.resnet101()
            model.load_state_dict({
                k[7:]: v
                for k, v in torch.load(
                    "/path/to/model"
                )['model_ema'].items()
                if "module" in k
            })
        else:
            raise ValueError(f"Unknown variant {variant}")

    elif name == "wrn50":
        if variant is None:
            model = models.wide_resnet50_2()
        elif variant == "v2a":
            model = models.wide_resnet50_2(weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        elif variant == "v2b":
            model = models.wide_resnet50_2()
            model.load_state_dict({
                k[7:]: v
                for k, v in torch.load(
                    "/path/to/model"
                )['model_ema'].items()
                if "module" in k
            })
        else:
            raise ValueError(f"Unknown variant {variant}")


    else:
        raise ValueError(f"Unknown model {model}")

    return model.eval()


