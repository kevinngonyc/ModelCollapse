import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from config import Paths, Labels, TrainingParams
import torch
import argparse
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
    mode = parts[0]
    label = Labels.mode_labels[mode]
    if mode in Labels.param_prefixes.keys():
        param = parts[1]
        append = Labels.param_prefixes[mode] + param
        return label + append
    else:
        return label

def get_generations(suffix):
    parts = suffix.split("_")
    if parts[-1][-3:] == "gen":
        return int(parts[-1][:-3])
    else:
        return TrainingParams.generations

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=Labels.viz_types, help="visualization type")
    parser.add_argument("suffix", help="suffix for output filename")
    parser.add_argument("-i", "--inputs", nargs="+", help="suffixes of input filenames")
    parser.add_argument("-g", "--generation", type=int, help="generation of sample")
    parser.add_argument("-y", "--yscale", help="y-scale of plot")
    args = parser.parse_args()

    if args.type == "single":
        generation = args.generation
        suffix = args.suffix
        input_path = os.path.join(Paths.tensor_dir, f"samples_{suffix}.pt")
        
        if suffix[-3:] == "gen":
            suffix = "_".join(suffix.split("_")[:-1])
        output_path = os.path.join(Paths.image_dir, f"gen{generation}_{suffix}.png")

        samples = torch.load(input_path, map_location=torch.device('cpu'))

        plt.subplots(figsize=(5, 5))
        show_image(make_grid(samples[generation],10,5))

        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

        plt.savefig(output_path)

    elif args.type == "loss_init":
        output_path = os.path.join(Paths.image_dir, f"loss_init_{args.suffix}.png")
        generations = get_generations(args.suffix)

        yscale = args.yscale if args.yscale else "linear"
        for i, suffix in enumerate(args.inputs):
            input_path = os.path.join(Paths.tensor_dir, f"loss_{suffix}.pt")
            loss = torch.load(input_path, map_location=torch.device('cpu'))
            if i == 0:
                baseline = loss[0].item()
                plt.plot([baseline for _ in range(generations + 1)], label="baseline", linestyle="dashed")
            plt.plot([l.item() for l in loss], label=label_suffix(suffix))

        plt.legend()
        plt.ylabel("Reconstruction Loss")
        plt.xlabel("Generation")
        plt.title("Generated Data Used as Input to Initial Model")
        plt.xticks(range(0, generations + 1, (generations + 10) // 20))
        plt.yscale(yscale)

        plt.savefig(output_path)

    print("Task complete.")