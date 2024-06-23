import multiprocessing as mp
import subprocess

def test_proc(arch, encoder, sparsity, title, dataset):
    process_args = ["python",
                    "test_pruning.py",
                    "--arch",
                    arch,
                    "--sparsity",
                    str(sparsity),
                    "--encoder",
                    encoder,
                    "--title",
                    title,
                    "--dataset",
                    dataset,
                    "--device",
                    "cpu",]
    subprocess.run(process_args)


if __name__ == "__main__":
    sparsity_list = [0.2,0.5,0.7,0.9]
    arch_list = ["fpn", "manet", "deeplabv3", "unet", "linknet", "unet++"]
    encoder = "mobilenet_v2"
    datasets = ["NEA", "CCSC"]
    title = ""
    args_list = [(arch, encoder, sparsity, title, dataset) for dataset in datasets for arch in arch_list for sparsity in sparsity_list]
    with mp.Pool(6) as pool:
       pool.starmap(test_proc, args_list)