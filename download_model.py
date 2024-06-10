import ssl

from setup import setup_model
from arguments import prepare_args

if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    args = prepare_args()
    model = setup_model(args)