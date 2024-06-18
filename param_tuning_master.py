import pathlib

import nni

from arguments import prepare_args


def start_param_tuning(args):
    experiment = nni.experiment.Experiment("local")
    command = f"python param_tuning_worker.py \
            --max-epoch {args.max_epoch} \
            --arch {args.arch} \
            --encoder {args.encoder} \
            --device {args.device}"
    experiment.config.trial_command = command
    trial_code_dir = pathlib.Path(".")/"trial_results"/args.title
    trial_code_dir.mkdir(parents=True, exist_ok=True)
    experiment.config.trial_code_directory = trial_code_dir.absolute()
    experiment.config.tuner.name = 'Evolution'
    # experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.tuner.class_args = {
        'optimize_mode': 'maximize',
        'population_size': 100
    }
    experiment.config.max_trial_number = 100
    experiment.config.trial_concurrency = 2
    search_space = {
        "lr": {"_type": "loguniform", "_value":[1e-5, 0.1]},
        "batch_size": {"_type":"choice", "_value":[2,4,8]},
        "optimizer_name": {"_type":"choice", "_value":["adamw","sgd","rmsprop"]}
    }
    experiment.config.search_space = search_space
    experiment.run(8009)
    
if __name__ == "__main__":
    args = prepare_args()
    start_param_tuning(args)