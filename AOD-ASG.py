import warnings
warnings.filterwarnings("ignore")

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('agg')
import numpy as np
import time
import os
import torch.utils.data
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from os.path import join as pjoin
from parser import parameter_parser
from load_data import split_ids, GraphData, collate_batch
from models.gcn import GCN
from sklearn import metrics




print('using torch', torch.__version__)
args = parameter_parser()
args.filters = list(map(int, args.filters.split(',')))
args.lr_decay_steps = list(map(int, args.lr_decay_steps.split(',')))
for arg in vars(args):
    print(arg, getattr(args, arg))

n_folds = args.folds
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
rnd_state = np.random.RandomState(args.seed)

print('Loading training_data...')

class DataReader():
    """
    Class to read the txt files containing all training_data of the dataset
    """
    def __init__(self, data_dir, rnd_state=None, use_cont_node_attr=False, folds=n_folds):
        self.data_dir = data_dir
        self.rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
        self.use_cont_node_attr = use_cont_node_attr
        files = os.listdir(self.data_dir)
        data = {}

        nodes, graphs, unique_id = self.read_graph_nodes_relations(
            list(filter(lambda f: f.find('graph_indicator') >= 0, files))[0])

        node_labels_file = list(filter(lambda f: f.find('node_labels') >= 0, files))
        if len(node_labels_file) == 1:
            data['features'] = self.read_node_features(node_labels_file[0], nodes, graphs, fn=lambda s: int(s.strip()))
        else:
            data['features'] = None

        data['adj_list'] = self.read_graph_adj(list(filter(lambda f: f.find('_A') >= 0, files))[0], nodes, graphs)
        data['targets'] = np.array(self.parse_txt_file(
            list(filter(lambda f: f.find('graph_labels') >= 0 or f.find('graph_attributes') >= 0, files))[0],
            line_parse_fn=lambda s: int(float(s.strip()))))

        data['ids'] = unique_id
        if self.use_cont_node_attr:
            data['attr'] = self.read_node_features(list(filter(lambda f: f.find('node_attributes') >= 0, files))[0],
                                                   nodes, graphs,
                                                   fn=lambda s: np.array(list(map(float, s.strip().split(',')))))
        features, n_edges, degrees = [], [], []
        for sample_id, adj in enumerate(data['adj_list']):
            N = len(adj)  # number of nodes
            if data['features'] is not None:
                assert N == len(data['features'][sample_id]), (N, len(data['features'][sample_id]))
            n = np.sum(adj)  # total sum of edges
            # assert n % 2 == 0, n
            n_edges.append(int(n / 2))  # undirected edges, so need to divide by 2
            # if not np.allclose(adj, adj.T):
            #     print(sample_id, 'not symmetric')
            degrees.extend(list(np.sum(adj, 1)))
            if data['features'] is not None:
                features.append(np.array(data['features'][sample_id]))

        # Create features over graphs as one-hot vectors for each node
            if data['features'] is not None:
                features_all = np.concatenate(features)
                features_min = features_all.min()
                num_features = int(features_all.max() - features_min + 1)  # number of possible values

        max_degree = np.max(degrees)

        features_onehot = []
        for sample_id, adj in enumerate(data['adj_list']):
            N = adj.shape[0]
            #node
            if data['features'] is not None:
                x = data['features'][sample_id]
                feature_onehot = np.zeros((len(x), num_features))
                for node, value in enumerate(x):
                    feature_onehot[node, value - features_min] = 1
            else:
                feature_onehot = np.empty((N, 0))
            #attr
            if self.use_cont_node_attr:
                feature_attr = np.array(data['attr'][sample_id])
            else:
                feature_attr = np.empty((N, 0))
            #degree
            # if self.degree:
            #     if args.degree:
            #         max_degree = int(max_degree)
            #         degree_onehot = np.zeros((N, max_degree + 1))
            #         degree_onehot[np.arange(N), np.sum(adj, 1).astype(np.int32)] = 1
            #     else:
            #         degree_onehot = np.empty((N, 0))

            node_features = np.concatenate((feature_onehot, feature_attr), axis=1)
            if node_features.shape[1] == 0:
                # dummy features for datasets without node labels/attributes
                # node degree features can be used instead
                node_features = np.ones((N, 1))
            features_onehot.append(node_features)

        num_features = features_onehot[0].shape[1]

        shapes = [len(adj) for adj in data['adj_list']]
        labels = data['targets']  # graph class labels
        labels -= np.min(labels)  # to start from 0

        classes = np.unique(labels)
        num_classes = len(classes)

        if not np.all(np.diff(classes) == 1):
            print('making labels sequential, otherwise pytorch might crash')
            labels_new = np.zeros(labels.shape, dtype=labels.dtype) - 1
            for lbl in range(num_classes):
                labels_new[labels == classes[lbl]] = lbl
            labels = labels_new
            classes = np.unique(labels)
            assert len(np.unique(labels)) == num_classes, np.unique(labels)

        def stats(x):
            return (np.mean(x), np.std(x), np.min(x), np.max(x))

        print('N nodes avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(shapes))
        print('N edges avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(n_edges))
        print('Node degree avg/std/min/max: \t%.2f/%.2f/%d/%d' % stats(degrees))
        print('Node features dim: \t\t%d' % num_features)
        print('N classes: \t\t\t%d' % num_classes)
        print('Classes: \t\t\t%s' % str(classes))
        for lbl in classes:
            print('Class %d: \t\t\t%d samples' % (lbl, np.sum(labels == lbl)))

        if data['features'] is not None:
            for u in np.unique(features_all):
                print('feature {}, count {}/{}'.format(u, np.count_nonzero(features_all == u), len(features_all)))

        N_graphs = len(labels)  # number of samples (graphs) in data
        assert len(data['adj_list']) == len(features_onehot), 'invalid data'
        assert N_graphs == len(data['adj_list']), 'invalid data'

        # Create train/test sets first
        train_ids, test_ids = split_ids(rnd_state.permutation(N_graphs), folds=folds)

        # Create train sets
        splits = []
        for fold in range(len(train_ids)):
            splits.append({'train': train_ids[fold],
                           'test': test_ids[fold]})

        data['features_onehot'] = features_onehot
        data['targets'] = labels
        data['splits'] = splits
        data['N_nodes_max'] = np.max(shapes)  # max number of nodes
        data['num_features'] = num_features
        data['num_classes'] = num_classes

        self.data = data

    def parse_txt_file(self, fpath, line_parse_fn=None):
        with open(pjoin(self.data_dir, fpath), 'r') as f:
            lines = f.readlines()
        data = [line_parse_fn(s) if line_parse_fn is not None else s for s in lines]
        return data

    def read_graph_adj(self, fpath, nodes, graphs):
        edges = self.parse_txt_file(fpath, line_parse_fn=lambda s: s.split(','))
        adj_dict = {}
        for edge in edges:
            # node1 = int(edge[0].strip()) - 1  # -1 because of zero-indexing in our code
            # node2 = int(edge[1].strip()) - 1
            node1 = int(edge[0].strip())
            node2 = int(edge[1].strip())
            graph_id = nodes[node1]
            assert graph_id == nodes[node2], ('invalid data', graph_id, nodes[node2])
            if graph_id not in adj_dict:
                n = len(graphs[graph_id])
                adj_dict[graph_id] = np.zeros((n, n))
            ind1 = np.where(graphs[graph_id] == node1)[0]
            ind2 = np.where(graphs[graph_id] == node2)[0]
            assert len(ind1) == len(ind2) == 1, (ind1, ind2)
            adj_dict[graph_id][ind1, ind2] = 1
        adj_list = [adj_dict[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return adj_list

    def read_graph_nodes_relations(self, fpath):
        graph_ids = self.parse_txt_file(fpath, line_parse_fn=lambda s: int(s.rstrip()))
        nodes, graphs = {}, {}
        for node_id, graph_id in enumerate(graph_ids):
            if graph_id not in graphs:
                graphs[graph_id] = []
            graphs[graph_id].append(node_id)
            nodes[node_id] = graph_id
        graph_ids = np.unique(list(graphs.keys()))
        unique_id = graph_ids
        for graph_id in graph_ids:
            graphs[graph_id] = np.array(graphs[graph_id])
        return nodes, graphs, unique_id

    def read_node_features(self, fpath, nodes, graphs, fn):
        node_features_all = self.parse_txt_file(fpath, line_parse_fn=fn)
        node_features = {}
        for node_id, x in enumerate(node_features_all):
            graph_id = nodes[node_id]
            if graph_id not in node_features:
                node_features[graph_id] = [None] * len(graphs[graph_id])
            ind = np.where(graphs[graph_id] == node_id)[0]
            assert len(ind) == 1, ind
            assert node_features[graph_id][ind[0]] is None, node_features[graph_id][ind[0]]
            node_features[graph_id][ind[0]] = x
        node_features_lst = [node_features[graph_id] for graph_id in sorted(list(graphs.keys()))]
        return node_features_lst

datareader = DataReader(data_dir='./data/%s/' % args.dataset, rnd_state=rnd_state,
                        use_cont_node_attr=args.use_cont_node_attr, folds=args.folds)

# train and test
result_folds = []

for fold_id in range(n_folds):
    loaders = []
    for split in ['train', 'test']:
        gdata = GraphData(fold_id=fold_id, datareader=datareader, split=split)
        loader = DataLoader(gdata, batch_size=args.batch_size, shuffle=split.find('train') >= 0,
                            num_workers=args.threads, collate_fn=collate_batch)
        loaders.append(loader)
    print('FOLD {}, train {}, test {}'.format(fold_id, len(loaders[0].dataset), len(loaders[1].dataset)))

    if args.model == 'gcn':
        model = GCN(in_features=loaders[0].dataset.num_features,
                    out_features=loaders[0].dataset.num_classes,
                    n_hidden=args.n_hidden,
                    filters=args.filters,
                    K=args.filter_scale,
                    bnorm=args.bn,
                    dropout=args.dropout,
                    adj_sq=args.adj_sq,
                    scale_identity=args.scale_identity).to(args.device)
    else:
        raise NotImplementedError(args.model)

    print('Initialize model...')

    train_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    print('N trainable parameters:', np.sum([p.numel() for p in train_params]))
    optimizer = optim.Adam(train_params, lr=args.lr, betas=(0.5, 0.999), weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, args.lr_decay_steps, gamma=0.1)  # dynamic adjustment lr
    # loss_fn = F.nll_loss  # when model is gcn_origin or gat, use this
    loss_fn = F.cross_entropy  # when model is gcn_modify, use this


    def train(train_loader):
        scheduler.step()
        model.train()
        start = time.time()
        train_loss, n_samples = 0, 0
        for batch_idx, data in enumerate(train_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            optimizer.zero_grad()
            # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
            output = model(data)  # when model is gcn_modify, use this
            loss = loss_fn(output, data[4])
            loss.backward()
            optimizer.step()
            time_iter = time.time() - start
            train_loss += loss.item() * len(output)
            n_samples += len(output)

        print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f} (avg: {:.6f}),  sec/iter: {:.4f}'.format(
            epoch + 1, n_samples, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
            loss.item(), train_loss / n_samples, time_iter / (batch_idx + 1)))

    def test(test_loader):
        model.eval()
        start = time.time()
        test_loss, n_samples, count = 0, 0, 0
        tn, fp, fn, tp = 0, 0, 0, 0  # calculate recall, precision, F1 score
        accuracy, recall, precision, F1, auc = 0, 0, 0, 0, 0
        fn_list = []  # Store the contract id corresponding to the fn
        fp_list = []  # Store the contract id corresponding to the fp
        tp_list = []  # Store the contract id corresponding to the tp
        tn_list = []  # Store the contract id corresponding to the tn
        for batch_idx, data in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(args.device)
            # output = model(training_data[0], training_data[1])  # when model is gcn_origin or gat, use this
            output = model(data)  # when model is gcn_modify, use this
            loss = loss_fn(output, data[4], reduction='sum')
            test_loss += loss.item()
            n_samples += len(output)
            pred = output.detach().cpu().max(1, keepdim=True)[1]
            count += 1

            for k in range(len(pred)):
                if (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 1):
                    # TP predict == 1 & label == 1
                    tp += 1
                    tp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 0):
                    # TN predict == 0 & label == 0
                    tn += 1
                    tn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 0) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 1):
                    # FN predict == 0 & label == 1
                    fn += 1
                    fn_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue
                elif (np.array(pred.view_as(data[4])[k]).tolist() == 1) & (
                        np.array(data[4].detach().cpu()[k]).tolist() == 0):
                    # FP predict == 1 & label == 0
                    fp += 1
                    fp_list.append(np.array(data[5].detach().cpu()[k]).tolist())
                    continue

            accuracy += metrics.accuracy_score(data[4].cpu(), pred.view_as(data[4]))
            recall += metrics.recall_score(data[4].cpu(), pred.view_as(data[4]))
            precision += metrics.precision_score(data[4].cpu(), pred.view_as(data[4]))
            F1 += metrics.f1_score(data[4].cpu(), pred.view_as(data[4]))
            try:
                auc += metrics.roc_auc_score(data[4].cpu(), pred.view_as(data[4]))
            except ValueError:
                pass
            #fpr, tpr, thersholds = metrics.roc_curve(data[4], pred.view_as(data[4], pos_label=1))

            fpr, tpr, thersholds = metrics.roc_curve(data[4].cpu(), pred.view_as(data[4]))
            for i, value in enumerate(thersholds):
                print("%f %f %f" % (fpr[i], tpr[i], value))

            # roc_auc = metrics.auc(fpr, tpr)
            # plt.figure()
            # lw = 2
            # plt.figure(figsize=(10, 10))
            # plt.plot(fpr, tpr, color='darkorange',
            #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)  ###假正率为横坐标，真正率为纵坐标做曲线
            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            # plt.xlim([0.0, 1.0])
            # plt.ylim([0.0, 1.05])
            # plt.xlabel('False Positive Rate')
            # plt.ylabel('True Positive Rate')
            # plt.title('Receiver operating characteristic curve (ROC)')
            # plt.legend(loc="lower right")
            # plt.show()

        print(tp, fp, tn, fn)
        accuracy = 100. * accuracy / count
        recall = 100. * recall / count
        precision = 100. * precision / count
        F1 = 100. * F1 / count
        auc = 100. * auc / count
        FPR = fp / (fp + tn)
        TPR = tp / (tp + fn) #recall(DR)

        print(
            'Test set (epoch {}): Average loss: {:.4f}, Accuracy: ({:.2f}%), Recall: ({:.2f}%), Precision: ({:.2f}%), '
            'F1-Score: ({:.2f}%), FPR: ({:.2f}%), TPR: ({:.2f}%),auc: ({:.2f}%), sec/iter: {:.4f}\n'.format(
                epoch + 1, test_loss / n_samples, accuracy, recall, precision, F1, FPR, TPR, auc,
                (time.time() - start) / len(test_loader))
        )

        # print("fn_list(predict == 0 & label == 1):", fn_list)
        # print("fp_list(predict == 1 & label == 0):", fp_list)
        # print("tn_list(predict == 0 & label == 1):", tn_list)
        # print("tp_list(predict == 1 & label == 0):", tp_list)
        print()

        return accuracy, recall, precision, F1, FPR, TPR, auc


    # if __name__ == '__main__':
    for epoch in range(args.epochs):
        train(loaders[0])
        accuracy, recall, precision, F1, FPR, TPR, auc = test(loaders[1])
    result_folds.append([accuracy, recall, precision, F1, FPR, TPR, auc])



print(result_folds)
acc_list = []
recall_list = []
precision_list = []
F1_list = []
FPR_list = []
TPR_list = []
auc_list = []

for i in range(len(result_folds)):
    acc_list.append(result_folds[i][0])
    recall_list.append(result_folds[i][1])
    precision_list.append(result_folds[i][2])
    F1_list.append(result_folds[i][3])
    FPR_list.append(result_folds[i][4])
    TPR_list.append(result_folds[i][5])
    auc_list.append(result_folds[i][6])


print(
    '{}-fold cross validation avg acc (+- std): {:.2f}% ({:.2f}%), recall (+- std): {:.2f}% ({:.2f}%), precision (+- std): {:.2f}% ({:.2f}%), '
    'F1-Score (+- std): {:.2f}% ({:.2f}%), FPR (+- fpr): {:.2f}% ({:.2f}%), TPR (+- tpr): {:.2f}% ({:.2f}%), AUC (+- auc): {:.2f}% ({:.2f}%)'.format(
        n_folds, np.mean(acc_list), np.std(acc_list), np.mean(recall_list), np.std(recall_list),
        np.mean(precision_list), np.std(precision_list), np.mean(F1_list), np.std(F1_list), np.mean(FPR_list),
        np.std(FPR_list),np.mean(TPR_list),np.std(TPR_list),np.mean(auc_list),np.std(auc_list)
    )
)




