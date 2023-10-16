import json

import numpy as np
import torch
import torch.nn as nn
from dgllife.utils import (
    EarlyStopping,
    PretrainAtomFeaturizer,
    PretrainBondFeaturizer,
    RandomSplitter,
)
from prettytable import PrettyTable
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_split import *
from dataset import *
from model import *

BATCH_SIZE = 256
K_FOLD = 5


def compute_metrics(y_true, y_prob):
    y_pred = np.array(y_prob) > 0.5
    BACC = balanced_accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    ACC = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred, average='binary')
    Prec = precision_score(y_true, y_pred, average='binary')
    Rec = recall_score(y_true, y_pred, average='binary')
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    ap = average_precision_score(y_true, y_prob)

    return ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap


def init_featurizer():
    args = {
        'node_featurizer': PretrainAtomFeaturizer(),
        'edge_featurizer': PretrainBondFeaturizer(),
    }
    return args


def get_dataloaders(dataset_name, data_path, config_drug_feature):
    label_filenames = data_path + dataset_name + '/labeled_triples_clean.csv'
    smiles_filenames = data_path + dataset_name + '/drug_set.json'
    context_filenames = data_path + dataset_name + '/context_set.json'
    label_df = pd.read_csv(label_filenames)
    drug1 = label_df['drug_1'].values
    drug2 = label_df['drug_2'].values
    context = label_df['context'].values
    labels = label_df['label'].values
    drug2smiles = {}
    with open(smiles_filenames, "r") as read_file:
        dict_d2s = json.load(read_file)
        for key, value in dict_d2s.items():
            drug2smiles[key] = value['smiles']

    context2features = {}
    with open(context_filenames, "r") as read_file:
        context2features = json.load(read_file)

    train_loaders = []
    valid_loaders = []
    test_loaders = []
    dataset = SeqDataset(
        drug1,
        drug2,
        drug2smiles,
        context,
        context2features,
        labels,
        dataset_name,
        config_drug_feature,
    )

    for i in range(K_FOLD):
        train_set, val_set, test_set = RandomSplitter.train_val_test_split(
            dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=i
        )

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )
        test_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )
        valid_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)

    return train_loaders, valid_loaders, test_loaders


def get_split_data_loader(dataset_name, split_scheme):
    label_filenames = data_path + dataset_name + '/labeled_triples_clean.csv'
    smiles_filenames = data_path + dataset_name + '/drug_set.json'
    context_filenames = data_path + dataset_name + '/context_set.json'

    drug2smiles = {}
    with open(smiles_filenames, "r") as read_file:
        dict_d2s = json.load(read_file)
        for key, value in dict_d2s.items():
            drug2smiles[key] = value['smiles']

    context2features = {}
    with open(context_filenames, "r") as read_file:
        context2features = json.load(read_file)

    train_loaders = []
    valid_loaders = []
    test_loaders = []
    if split_scheme == 'random':
        label_df = pd.read_csv(label_filenames)
        drug1 = label_df['drug_1'].values
        drug2 = label_df['drug_2'].values
        context = label_df['context'].values
        labels = label_df['label'].values

        dataset = SeqDataset(
            drug1,
            drug2,
            drug2smiles,
            context,
            context2features,
            labels,
            dataset_name,
            config_drug_feature,
        )

    for i in range(K_FOLD):
        if split_scheme == 'random':
            train_set, val_set, test_set = RandomSplitter.train_val_test_split(
                dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1, random_state=i
            )
        else:
            if split_scheme == 'cold_drug':
                train_data, valid_data, test_data = cold_drug_split(label_filenames, random_state=i)
            elif split_scheme == 'cold_cell':
                train_data, valid_data, test_data = cold_cell_split(label_filenames, random_state=i)
            elif split_scheme == 'cold_drugs':
                train_data, valid_data, test_data = cold_drugs_split(
                    label_filenames, random_state=i
                )
            elif split_scheme == 'both_cold':
                train_data, valid_data, test_data = both_cold_split(label_filenames, random_state=i)

            train_set = SeqDataset(
                train_data['drug_1'].to_list(),
                train_data['drug_2'].to_list(),
                drug2smiles,
                train_data['context'].to_list(),
                context2features,
                train_data['label'].to_list(),
                dataset_name,
                config_drug_feature,
            )
            val_set = SeqDataset(
                valid_data['drug_1'].to_list(),
                valid_data['drug_2'].to_list(),
                drug2smiles,
                valid_data['context'].to_list(),
                context2features,
                valid_data['label'].to_list(),
                dataset_name,
                config_drug_feature,
            )
            test_set = SeqDataset(
                test_data['drug_1'].to_list(),
                test_data['drug_2'].to_list(),
                drug2smiles,
                test_data['context'].to_list(),
                context2features,
                test_data['label'].to_list(),
                dataset_name,
                config_drug_feature,
            )

        train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )
        test_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )
        valid_loader = DataLoader(
            test_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_molgraphs,
        )

        train_loaders.append(train_loader)
        valid_loaders.append(valid_loader)
        test_loaders.append(test_loader)

    return train_loaders, valid_loaders, test_loaders


def run_a_train_epoch(
    device, epoch, num_epochs, model, data_loader, loss_criterion, optimizer, scheduler
):
    model.train()
    tbar = tqdm(enumerate(data_loader), total=len(data_loader))

    for id, (*x, y) in tbar:
        for i in range(len(x)):
            x[i] = x[i].to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(*x)

        loss = loss_criterion(output.view(-1), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        tbar.set_description(f' * Train Epoch {epoch} Loss={loss.item()  :.3f}')


def run_an_eval_epoch(model, data_loader):
    model.eval()
    with torch.no_grad():
        preds = torch.Tensor()
        trues = torch.Tensor()
        for batch_id, (*x, y) in tqdm(enumerate(data_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            logits = model(*x)
            preds = torch.cat((preds, logits.cpu()), 0)
            trues = torch.cat((trues, y.view(-1, 1).cpu()), 0)

        preds, trues = preds.numpy().flatten(), trues.numpy().flatten()
    return preds, trues


data_path = './dataset/'

lr = 3e-3
num_epochs = 100

dataset_name = 'drugcomb'  # 'drugcomb', 'drugcombdb'
split_scheme = 'random'  # 'random', 'cold_drug', 'cold_cell', 'cold_drugs', 'both_cold'

if __name__ == '__main__':
    t_tables = PrettyTable(
        ['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap']
    )
    t_tables.float_format = '.3'
    test_ACC = []
    test_BACC = []
    test_Prec = []
    test_Rec = []
    test_F1 = []
    test_roc_auc = []
    test_mcc = []
    test_kappa = []
    test_ap = []

    config_drug_feature = init_featurizer()

    train_loaders, valid_loaders, test_loaders = get_split_data_loader(dataset_name, split_scheme)

    for i in range(K_FOLD):
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        model = DeepGraphFusion(cell_dim=18046).to(device)

        optimizer = optim.AdamW(model.parameters())
        loss_criterion = nn.BCELoss()
        stopper = EarlyStopping(mode='higher', patience=15, filename='results/tri_fusion_fp')
        all_tables = PrettyTable(
            ['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap']
        )
        all_tables.float_format = '.3'

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loaders[i])
        )
        for epoch in range(num_epochs):
            # Train
            run_a_train_epoch(
                device,
                epoch,
                num_epochs,
                model,
                train_loaders[i],
                loss_criterion,
                optimizer,
                scheduler,
            )

            # Validation and early stop
            val_pred, val_true = run_an_eval_epoch(model, valid_loaders[i])
            ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = compute_metrics(val_true, val_pred)
            early_stop = stopper.step(ACC, model)

            e_tables = PrettyTable(
                ['epoch', 'ACC', 'BACC', 'Prec', 'Rec', 'F1', 'AUC', 'MCC', 'kappa', 'ap']
            )
            e_tables.float_format = '.3'
            row = [epoch, ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
            e_tables.add_row(row)
            all_tables.add_row(row)
            print(e_tables)

            if early_stop:
                break

        stopper.load_checkpoint(model)
        test_pred, test_y = run_an_eval_epoch(model, test_loaders[i])

        ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap = compute_metrics(test_y, test_pred)
        row = ['test', ACC, BACC, Prec, Rec, F1, roc_auc, mcc, kappa, ap]
        all_tables.add_row(row)
        t_tables.add_row(row)
        print(all_tables)

        test_ACC.append(ACC)
        test_BACC.append(BACC)
        test_Prec.append(Prec)
        test_Rec.append(Rec)
        test_F1.append(F1)
        test_roc_auc.append(roc_auc)
        test_mcc.append(mcc)
        test_kappa.append(kappa)
        test_ap.append(ap)

    row = [
        'mean',
        np.mean(test_ACC),
        np.mean(test_BACC),
        np.mean(test_Prec),
        np.mean(test_Rec),
        np.mean(test_F1),
        np.mean(test_roc_auc),
        np.mean(test_mcc),
        np.mean(test_kappa),
        np.mean(test_ap),
    ]
    t_tables.add_row(row)

    row = [
        'std',
        np.std(test_ACC),
        np.std(test_BACC),
        np.std(test_Prec),
        np.std(test_Rec),
        np.std(test_F1),
        np.std(test_roc_auc),
        np.std(test_mcc),
        np.std(test_kappa),
        np.std(test_ap),
    ]
    t_tables.add_row(row)

    print(t_tables)
