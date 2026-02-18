# ==========================================
# DATA LOADING
# ==========================================

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset, DataLoader
from ..config import config_by_ntip, SEED, BATCH_SIZE, BASE_DATA_PATH


def get_loaders(ntip):
    """
    Loads phylogenetic tree data (F-matrices) and beta-splitting labels,
    normalizes them, and creates stratified DataLoaders.

    Assumes data is stored in a specific directory structure with
    5 distinct beta-splitting scenarios (different tree shapes).

    Args:
        ntip (int): Number of tips (taxa) in the trees. Used to select the
                    correct dataset and normalize input values.

    Returns:
        tuple: (train_loader, val_loader, test_loader, input_dim)
               - Loaders allow iteration over shuffled batches.
               - input_dim is the size of the flattened feature vector.
    """

    n_scenarios = 5   # simulate from beta-splitting model with 5 different betas
    config = config_by_ntip[ntip]
    samples_per_scenario = config['samples_per_scenario']
    base_path = f"{BASE_DATA_PATH}BF_sim_betaSplit_ntip{ntip}_eachBeta{int(samples_per_scenario/1000)}k/"
    data_path = base_path + "Fmat_below_sub.csv"
    label_path = base_path + "beta.csv"

    print(f"Loading data from: {data_path}")
    print(f"Loading labels from: {label_path}")

    # Load data
    df_X = pd.read_csv(data_path)
    X = torch.tensor(df_X.values, dtype=torch.float32)  # X is a torch.FloatTensor
    X = X / ntip  # normalize to [0,1]
    input_dim = X.shape[1]

    # Load labels (betas)
    df_y = pd.read_csv(label_path)
    y = torch.tensor(df_y.values, dtype=torch.float32).view(-1)

    full_dataset = TensorDataset(X, y)

    split_ratio = config['split']
    train_frac = split_ratio[0]
    val_frac = split_ratio[1]

    # stratified
    train_per_scenario = int(train_frac * samples_per_scenario)
    val_per_scenario = int(val_frac * samples_per_scenario)
    test_per_scenario = samples_per_scenario - train_per_scenario - val_per_scenario

    rng = np.random.default_rng(seed=SEED)
    train_indices = []
    val_indices = []
    test_indices = []

    for s in range(n_scenarios):
        start = s * samples_per_scenario
        end = (s + 1) * samples_per_scenario
        indices = np.arange(start, end)

        rng.shuffle(indices)

        train_indices.extend(indices[:train_per_scenario])
        val_indices.extend(indices[train_per_scenario : train_per_scenario + val_per_scenario])
        test_indices.extend(indices[train_per_scenario + val_per_scenario:])

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    test_dataset = Subset(full_dataset, test_indices)

    print(f"Total Split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, input_dim
