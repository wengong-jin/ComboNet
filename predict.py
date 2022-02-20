import os, random, sys
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F

from covid_train import DiseaseModel
from sklearn.metrics import roc_auc_score, classification_report


def combo_evaluate(model, data, args):
    model.eval()
    all_preds = []
    for i in range(0, len(data), args.batch_size):
        mol_batch = data[i : i + args.batch_size]
        smiles1, smiles2 = list(zip(*mol_batch))[:2]
        preds = model.combo_forward(smiles1, smiles2, mode=0).cpu().numpy()
        all_preds.append(preds)
    return np.concatenate(all_preds, axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    args = parser.parse_args()

    with open(args.test_path) as f:
        header = next(f)
        data = [line.strip("\r\n ").split(',')[:2] for line in f]

    args.checkpoint_paths = []
    for root, _, files in os.walk(args.checkpoint_dir):
        for fname in files:
            if fname.endswith('.pt'):
                args.checkpoint_paths.append(os.path.join(root, fname))

    sum_preds = np.zeros((len(data),))
    with torch.no_grad():
        for checkpoint_path in args.checkpoint_paths:
            ckpt = torch.load(checkpoint_path)
            ckpt['args'].attention = False
            model = DiseaseModel(ckpt['args']).cuda()
            model.load_state_dict(ckpt['state_dict'])
            model_preds = combo_evaluate(model, data, ckpt['args'])
            sum_preds += np.array(model_preds)[:, 0]

    sum_preds += 10
    print("row_smiles,col_smiles,score")
    for (x, y), score in zip(data, sum_preds):
        print(f"{x},{y},{score:.6f}")

