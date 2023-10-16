import random
from collections import defaultdict
from typing import ClassVar, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def cold_drugs_split(label_files, random_state):
    data = pd.read_csv(label_files)
    columns: ClassVar[Sequence[str]] = ("drug_1", "drug_2", "context", "label")
    dtype: ClassVar[Mapping[str, type]] = {
        "drug_1": str,
        "drug_2": str,
        "context": str,
        "label": float,
    }

    unique_drug_pairs = defaultdict(list)
    for index, row in data.iterrows():
        unique_drug_pairs[(row['drug_1'], row['drug_2'])].append(
            [row['drug_1'], row['drug_2'], row['context'], row['label']]
        )

    train_drugpair_size = int(len(unique_drug_pairs) * 0.8)
    train_drugpairs = random.sample(unique_drug_pairs.keys(), train_drugpair_size)
    train_valid_list = []
    for key in train_drugpairs:
        for row in unique_drug_pairs[key]:
            train_valid_list.append(row)
    train_valid_data = pd.DataFrame(train_valid_list, columns=columns).astype(dtype)

    train_data, valid_data = train_test_split(
        train_valid_data, train_size=0.75, random_state=random_state
    )

    test_drugpairs = list(set(unique_drug_pairs.keys()) - set(train_drugpairs))

    test_list = []
    for key in test_drugpairs:
        for row in unique_drug_pairs[key]:
            test_list.append(row)
    test_data = pd.DataFrame(test_list, columns=columns).astype(dtype)

    return train_data, valid_data, test_data


def cold_drug_split(label_files, random_state):
    data = pd.read_csv(label_files)
    columns: ClassVar[Sequence[str]] = ("drug_1", "drug_2", "context", "label")
    dtype: ClassVar[Mapping[str, type]] = {
        "drug_1": str,
        "drug_2": str,
        "context": str,
        "label": float,
    }
    unique_drugs = pd.unique(data[['drug_1', 'drug_2']].values.ravel())
    train_drug_size = int(unique_drugs.shape[0] * 0.8)
    train_drugs = np.random.choice(unique_drugs, train_drug_size, replace=False)
    test_drugs = list(set(list(unique_drugs)) - set(list(train_drugs)))

    train_valid_data = data[data[['drug_1', 'drug_2']].isin(train_drugs).all(axis=1)]
    test_data = data[data[['drug_1', 'drug_2']].isin(test_drugs).all(axis=1)]

    train_data, valid_data = train_test_split(
        train_valid_data, train_size=0.75, random_state=random_state
    )

    return train_data, valid_data, test_data


def cold_cell_split(label_files, random_state):
    data = pd.read_csv(label_files)
    columns: ClassVar[Sequence[str]] = ("drug_1", "drug_2", "context", "label")
    dtype: ClassVar[Mapping[str, type]] = {
        "drug_1": str,
        "drug_2": str,
        "context": str,
        "label": float,
    }
    unique_cells = pd.unique(data[['context']].values.ravel())

    train_cell_size = int(unique_cells.shape[0] * 0.8)
    train_cells = np.random.choice(unique_cells, train_cell_size, replace=False)
    test_cells = list(set(list(unique_cells)) - set(list(train_cells)))

    train_valid_data = data[data[['context']].isin(train_cells).all(axis=1)]
    test_data = data[data[['context']].isin(test_cells).all(axis=1)]

    train_data, valid_data = train_test_split(
        train_valid_data, train_size=0.75, random_state=random_state
    )

    return train_data, valid_data, test_data


def both_cold_split(label_files, random_state):
    data = pd.read_csv(label_files)
    unique_cells = pd.unique(data[['context']].values.ravel())
    unique_drugs = pd.unique(data[['drug_1', 'drug_2']].values.ravel())

    np.random.shuffle(unique_cells)
    np.random.shuffle(unique_drugs)
    train_drug_size = int(len(unique_drugs) * 0.8)
    train_cell_size = int(len(unique_cells) * 0.8)
    train_valid_samples = []
    test_samples = []
    samples_matrix = np.zeros(
        (unique_drugs.shape[0], unique_drugs.shape[0], unique_cells.shape[0])
    )  # a binay matrix: drug * drug * cell
    for index, row in tqdm(data.iterrows()):
        i = np.where(unique_drugs == row['drug_1'])[0][0]
        j = np.where(unique_drugs == row['drug_2'])[0][0]
        k = np.where(unique_cells == row['context'])[0][0]
        samples_matrix[i][j][k] = 1
        if i < train_drug_size and j < train_drug_size and k < train_cell_size:
            train_valid_samples.append([row['drug_1'], row['drug_2'], row['context'], row['label']])
        if i >= train_drug_size and j >= train_drug_size and k >= train_cell_size:
            test_samples.append([row['drug_1'], row['drug_2'], row['context'], row['label']])

    columns: ClassVar[Sequence[str]] = ("drug_1", "drug_2", "context", "label")
    dtype: ClassVar[Mapping[str, type]] = {
        "drug_1": str,
        "drug_2": str,
        "context": str,
        "label": float,
    }

    train_valid_data = pd.DataFrame(train_valid_samples, columns=columns)

    train_data, valid_data = train_test_split(
        train_valid_data, train_size=0.75, random_state=random_state
    )
    test_data = pd.DataFrame(test_samples, columns=columns)

    unique_cells_train = pd.unique(train_data[['context']].values.ravel())
    unique_drugs_train = pd.unique(train_data[['drug_1', 'drug_2']].values.ravel())

    unique_cells_test = pd.unique(test_data[['context']].values.ravel())
    unique_drugs_test = pd.unique(test_data[['drug_1', 'drug_2']].values.ravel())

    print(
        unique_cells_train.shape,
        unique_drugs_train.shape,
        unique_cells_test.shape,
        unique_drugs_test.shape,
    )
    print(np.intersect1d(unique_cells_test, unique_cells_train))
    print(np.intersect1d(unique_drugs_train, unique_drugs_test))
    return train_data, valid_data, test_data
