import argparse

def parameter_parser():
    # Experiment parameters
    parser = argparse.ArgumentParser(description='User behavior anomaly detection based on graph neural network')
    parser.add_argument('-D', '--dataset', type=str, default='UBA1_0',
                        choices=['UBA1_0', 'UBA1_1g', 'UBA1_2s','UBA1_3ss','UBA2_0', 'UBA2_1g', 'UBA2_2s','UBA2_3ss',
                                 'UBA3_0', 'UBA3_1g', 'UBA3_2s','UBA3_3ss','demo'])
    parser.add_argument('-M', '--model', type=str, default='gcn')
    parser.add_argument('--lr', type=float, default=0.02, help='learning rate')
    parser.add_argument('--lr_decay_steps', type=str, default='1,3', help='learning rate')
    parser.add_argument('--wd', type=float, default= 1e-4, help='weight decay')
    parser.add_argument('-d', '--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('-f', '--filters', type=str, default='64,64,64', help='number of filters in each layer')
    parser.add_argument('-K', '--filter_scale', type=int, default=1,
                        help='filter scale (receptive field size), must be > 0; 1 for GCN, >1 for ChebNet')
    parser.add_argument('--n_hidden', type=int, default=256,
                        help='number of hidden units in a fully connected layer after the last conv layer')
    parser.add_argument('--degree', action='store_true', default=True, help='use one-hot node degree features')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--bn', action='store_true', default=False, help='use BatchNorm layer')
    parser.add_argument('-t', '--threads', type=int, default=0, help='number of threads to load training_data')
    parser.add_argument('--log_interval', type=int, default=1,
                        help='interval (number of batches) of logging')
    parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'])
    parser.add_argument('--seed', type=int, default=121, help='random seed')
    parser.add_argument('--shuffle_nodes', action='store_true', default=True, help='shuffle nodes for debugging')
    parser.add_argument('-F', '--folds', default=5, choices=[3, 5, 10], help='n-fold cross validation')
    parser.add_argument('-a', '--adj_sq', action='store_true', default=True,
                        help='use A^2 instead of A as an adjacency matrix')
    parser.add_argument('-s', '--scale_identity', action='store_true', default=False,
                        help='use 2I instead of I for self connections')
    parser.add_argument('-c', '--use_cont_node_attr', action='store_true', default=True,
                        help='use continuous node attributes in addition to discrete ones')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha value for the leaky_relu')
    parser.add_argument('--multi_head', type=int, default=4, help='number of head attentions(Multi-Head)')

    return parser.parse_args()