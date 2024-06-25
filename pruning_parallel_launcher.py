import multiprocessing as mp
import subprocess

def pruning_proc(arch, encoder, sparsity, dataset, pruner):
    # python pruning.py --arch fpn --dataset NEA --batch-size 4 --lr 3e-4 --max-epoch 100 --device cuda --encoder mobilenet_v2 --title exp_1 --sparsity 0.2

    process_args = ["python",
                    "pruning.py",
                    "--arch",
                    arch,
                    "--encoder",
                    encoder,
                    "--pruner",
                    pruner,
                    "--sparsity",
                    str(sparsity),
                    "--dataset",
                    dataset,
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
                    "3",
                    "--device",
                    "cuda",
                    "--max-epoch",
                    "150"]
    subprocess.run(process_args)


if __name__ == "__main__":
    import socket
    hostname = socket.gethostname()
    sparsity_list = [0.2, 0.5, 0.9]
    hostname_arch_list_dict = {
        "komputasi03":["fpn"],
        "komputasi04":["manet"],
        "komputasi06":["unet"],
        "komputasi07":["linknet"],
        "komputasi09":["unet++"],
    }
    arch_list = hostname_arch_list_dict[hostname]
    # arch_list = ["fpn", "manet", "deeplabv3", "unet", "linknet", "unet++"]
    encoder = "resnet101"
    dataset_list = ["CCSC", "NEA"]
    pruner_list = ["agp","linear","movement"]
    args_list = [(arch, encoder, sparsity, dataset, pruner) for dataset in dataset_list for pruner in pruner_list for arch in arch_list for sparsity in sparsity_list]
    # with mp.Pool(1) as pool:
    #    pool.starmap(pruning_proc, args_list)
    for args in args_list:
        pruning_proc(*args)