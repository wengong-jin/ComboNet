# Deep learning identifies synergistic drug combinations for treating COVID-19

This is the implementation of our PNAS 2021 paper: https://www.pnas.org/content/118/39/e2105070118

## Dependencies
Our model is tested in Linux with the following packages:
* CUDA >= 11.1
* PyTorch == 1.8.2 (LTS Version)
* Numpy >= 1.18.1
* tqdm

## Data

The covid combination data is stored in the `data/covid` folder.
* `data/covid/dti.csv` is the drug-target interaction data
* `data/covid/single_agent.csv` is the single-agent antiviral activity data
* `data/covid/synergy_train.csv` is the drug combination synergy data (training set)
* `data/covid/synergy_test.csv` is the drug combination synergy data (test set)
* `data/covid/synergy_test.csv` is the drug combination synergy data (test set under "compounds out" strategy)
* `data/covid/synergy_experiment.csv` contains the top 30 drug combinations ranked by ComboNet and we experimentally tested them in a VeroE6 CPE assay.


## Model training

To run our model under five-fold cross-validation, please run
```
python covid_train.py --save_dir ckpts/combonet
```

