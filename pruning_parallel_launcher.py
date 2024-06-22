import multiprocessing as mp
import subprocess

def pruning_proc(arch, encoder, sparsity):
    # python pruning.py --arch fpn --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2

    process_args = ["python",
                    "pruning.py",
                    "--arch",
                    arch,
                    "--encoder",
                    encoder,
                    "--sparsity",
                    str(sparsity),
                    "--dataset",
                    "NEA",
                    "--batch-size",
                    "4",
                    "--lr",
                    "3e-4",
                    "--momentum",
                    "0.5",
                    "--optimizer-name",
                    "sgd",
                    "--loss-combination",
                    "tversky",
                    "--num-workers",
                    "4",
                    "--device",
                    "cuda",
                    "--max-epoch",
                    "100"]
    subprocess.run(process_args)


if __name__ == "__main__":
    sparsity_list = [0.2]
    arch_list = ["fpn", "manet", "deeplabv3", "unet", "linknet", "unet++"]
    encoder = "mobilenet_v2"
    args_list = [(arch, encoder, sparsity) for arch in arch_list for sparsity in sparsity_list]
    with mp.Pool(2) as pool:
       pool.starmap(pruning_proc, args_list)