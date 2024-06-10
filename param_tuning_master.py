from arguments import prepare_args
import nni


def start_param_tuning(args):
    experiment = nni.experiment.Experiment("local")
    command = f"python param_tuning_worker.py \
            --max-epoch {args.max_epoch} \
            --arch {args.arch} \
            --encoder {args.encoder} \
            --device {args.device}"
    experiment.config.trial_command = command
    experiment.config.trial_code_directory = "."
    experiment.config.tuner.name = 'TPE'
    experiment.config.tuner.class_args['optimize_mode'] = 'maximize'
    experiment.config.max_trial_number = 50
    experiment.config.trial_concurrency = 2
    search_space = {
        "lr": {"_type": "loguniform", "_value":[1e-5, 0.1]},
        "batch_size": {"_type":"choice", "_value":[2,4,8]}
    }
    experiment.config.search_space = search_space
    experiment.run(8080)
    
if __name__ == "__main__":
    args = prepare_args()
    
    start_param_tuning(args)