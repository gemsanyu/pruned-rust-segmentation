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
                        default="run-1",
                        help="title to differentiate between experiments")
    parser.add_argument('--arch',
                        choices=["fpn","unet","unet++","manet","linknet","pspnet","pan","deeplabv3","deeplabv3+"],
                        default="fpn",
                        help="choices for architecture")
    parser.add_argument('--dataset',
                        choices=["NEA"],
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
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="training learning rate")
    
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
    
    args = parser.parse_args(sys.argv[1:])
    args.title = args.arch+"_"+args.encoder+"_"+args.dataset+"_"+args.title
    return args
    