# Deep learning identifies synergistic drug combinations for treating COVID-19

This is the implementation of our PNAS 2021 paper: https://www.pnas.org/content/118/39/e2105070118

## Dependencies
Our model is tested in Linux with the following packages:
* CUDA >= 11.1
* PyTorch == 1.8.2 (LTS Version)
* Numpy >= 1.18.1
* tqdm

## Data

The covid combination data is stored in the `data/` folder.

## Model training

To run our model under five-fold cross-validation, please run
```
python covid_train.py --save_dir ckpts/combonet
```

