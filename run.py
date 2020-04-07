import argparse

import numpy as np

from utils.data import load_data, prepare_data, to_array
from utils.model import train_model, load_model, predict_model, latest_modified_weight
from utils.plot import model_plot


def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected')

    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str2bool, default=True,
                        help='True: Load trained model  False: Train model default: True')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    np.random.seed(0)
    args = parse_args()
    data = load_data('data/house_prices.csv')
    features = prepare_data(data, ('sqft_living', 'price'))
    x, y = to_array(features)
    if args.load:
        train_model(x, y)
    else:
        weight = latest_modified_weight()
        model = load_model(weight)
        model_plot(model, x, y)
        price = predict_model(model, 2000)
        print(price)
