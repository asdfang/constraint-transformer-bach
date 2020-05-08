# Augmentative Generation for Transformer Bach

## Usage
A conda environment is provided in `environment.yml`. Load it with `conda env create -f environment.yml`. The environment is named `tbach` and can be activated with `conda activate tbach`.

Every time we run `main.py`, we can do one of three things:
1. load a model with the `--load` flag
2. train a model with the `--aug_gen` to get the augmented model
3. train a model with the `--base` flag for the base model

Furthermore, we can generate from the final model with the `--generate` flag. 

On the first run, the dataset will need to be created in `data/`. Enter `y` and then `index` when prompted to create the most general vocabulary possible. After building the dataset, training should start. Models are saved in the `models/` folder. Generations are saved in the `models/model_id/generations` folder. 

To replicate our experiments, first set the desired `'savename'` and `'description'` for the model in `transformer_bach/bach_decoder_config.py`. The given description is saved in a `README.txt` stored in `model_dir`. Then, to train the augmented model, run
```
python main.py --aug_gen --config=transformer_bach/bach_decoder_config.py
```
To train the base model, run
```
python main.py --base --config=transformer_bach/bach_decoder_config.py
```

Throughout model training, `dataset_sizes.csv`, `grades.csv`, and `loss.csv` will be populated with useful information. To visualize this data, run
```
python experiments/training_plots.py --model_dir=models/model_id
```
The plots will be created in the `models/model_id/plots` folder.