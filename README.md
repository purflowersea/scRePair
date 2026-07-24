# scRePair: A Cell Relation Refinement and Paired Cross-omics Consistency Framework for Single-cell Multi-omics Clustering


# Installation & Dependencies


scRePair is mainly tested in a Linux environment, and its dependencies are listed below.


| Package         | Version  |
|-----------------|----------|
| python          | 3.9.21   |
| rdkit           | 2024.3.6 |
| pytorch         | 2.5.0    |
| cuda            | 12.4     |
| torch-geometric | 2.0.2    |
| torch-scatter   | 2.1.2    |
| torch-sparse    | 0.6.18   |
| torchvision     | 0.20.0   |
| scikit-learn    | 1.5.1    |
| tqdm            | 4.67.1   |
| networkx        | 3.2.1    |
| matplotlib      | 3.9.4    |
| pandas          | 2.2.3    |
| numpy           | 1.26.4   |


# Run seRePair


You can train seRePair with the following command:

## Step 1: Pre-training model

```bash
python main.py --name PBMC-3k --pretrain True
```

## Step 2: Formal training model with pre-training model

```bash
python main.py --name PBMC-3k
```




