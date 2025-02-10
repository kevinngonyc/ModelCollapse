import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from config import Paths, Labels
import torch
import sys
import os

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def label_suffix(suffix):
    parts = suffix.split("_")
    mod = parts[0]
    if mod in Labels.param_prefixes.keys():
        param = parts[1]
        label = Labels.mod_labels[mod]
        append = Labels.param_prefixes[mod] + param
        return label + append
    else:
        return Labels.mod_labels[suffix]

if __name__ == "__main__":

    if sys.argv[1] == "single":
        generation = int(sys.argv[2])
        suffix = sys.argv[3]
        input_path = os.path.join(Paths.tensor_dir, f"samples_{suffix}.pt")
        output_path = os.path.join(Paths.image_dir, f"gen{generation}_{suffix}.png")

        samples = torch.load(input_path, map_location=torch.device('cpu'))

        plt.subplots(figsize=(5, 5))
        show_image(make_grid(samples[generation],10,5))

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.savefig(output_path)

    elif sys.argv[1] == "loss_init":
        output_name = sys.argv[2]
        output_path = os.path.join(Paths.image_dir, f"loss_init_{output_name}.png")

        yscale = sys.argv[3]
        for i, suffix in enumerate(sys.argv[4:]):
            input_path = os.path.join(Paths.tensor_dir, f"loss_{suffix}.pt")
            loss = torch.load(input_path, map_location=torch.device('cpu'))
            if i == 0:
                baseline = loss[0].item()
                plt.plot([baseline for _ in range(21)], label="baseline", linestyle="dashed")
            plt.plot([l.item() for l in loss], label=label_suffix(suffix))

        plt.legend()
        plt.ylabel("Reconstruction Loss")
        plt.xlabel("Generation")
        plt.title("Generated Data Used as Input to Initial Model")
        plt.xticks(range(0, 21))
        plt.yscale(yscale)

        plt.savefig(output_path)

    print("Task complete.")