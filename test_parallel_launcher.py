import multiprocessing as mp
import subprocess

def test_proc(arch, encoder, title, dataset):
    process_args = ["python",
                    "test.py",
                    "--arch",
                    arch,
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
    arch_list = ["fpn", "manet", "deeplabv3", "unet", "linknet", "unet++"]
    encoder = "mobilenet_v2"
    title = ""
    datasets = ["NEA", "CCSC"]
    args_list = [(arch, encoder, title, dataset) for dataset in datasets for arch in arch_list]
    with mp.Pool(6) as pool:
       pool.starmap(test_proc, args_list)