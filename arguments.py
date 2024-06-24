import argparse
import sys

def prepare_args():
    parser = argparse.ArgumentParser(description='quantized-cnn-rust-segment')
    parser.add_argument('--seed',
                        type=int,
                        default=1,
                        help="rng seed")
    parser.add_argument('--device',
                        type=str,
                        default="cpu",
                        help="device: cuda or cpu")
    parser.add_argument('--title',
                        type=str,
                        default="",
                        help="title to differentiate between experiments")
    parser.add_argument('--arch',
                        choices=["fpn","unet","unet++","manet","linknet","pspnet","pan","deeplabv3","deeplabv3+"],
                        default="fpn",
                        help="choices for architecture")
    parser.add_argument('--dataset',
                        choices=["NEA", "CCSC"],
                        default="NEA",
                        help="name of dataset")
    
    
    # Training
    parser.add_argument('--batch-size',
                        type=int,
                        default=2,
                        help="training batch size")
    parser.add_argument('--max-epoch',
                        type=int,
                        default=10,
                        help="trai maximum num of epoch")
    parser.add_argument('--num-workers',
                        type=int,
                        default=0,
                        help="num workers for dataloaders")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="training learning rate")
    parser.add_argument('--momentum',
                        type=float,
                        default=0.5,
                        help="optimizer's momentum")
    parser.add_argument('--optimizer-name',
                        type=str,
                        default="sgd",
                        help="optimizer's momentum")
    parser.add_argument('--loss-combination',
                        type=str,
                        choices=["focal_dice", "focal_tversky", "tversky", "dice", "focal"],
                        default="focal_dice",
                        help="loss function")
    
    # Pretrained Encoder
    parser.add_argument('--encoder',
                        type=str,
                        default="se_resnext50_32x4d",
                        help="encoder name")
    parser.add_argument('--encoder-pretrained-source',
                        type=str,
                        default="imagenet",
                        help="pretrained source (name) for encoder")
    
    # for pruning
    parser.add_argument('--sparsity',
                        type=float,
                        default=0.2,
                        help="sparsity target for pruning")
    parser.add_argument('--pruner',
                        type=str,
                        choices=["agp","linear","movement"],
                        default="agp",
                        help="pruning algorithm")
    
    args = parser.parse_args(sys.argv[1:])
    args.title = args.arch+"_"+args.encoder+"_"+args.dataset+"_"+args.title
    return args
    