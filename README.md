# Augmentative Generation for Transformer Bach

## Usage
A conda environment is provided in `environment.yml`. Load it with `conda env create -f environment.yml`. The environment is named `tbach` and can be activated with `conda activate tbach`.

Every time we run `main.py`, we can do any combination of three things:
1. load a model with the `--load` flag
2. train a model with the `--train` flag
3. update a trained model with the `--update` flag

Furthermore, we can generate from the final model with the `--generate` flag. 

On the first run, the dataset will need to be created in `data/`. After building the dataset, training should start. Models are saved in the `models/` folder. Generations are saved in the `models/model_id/generations` folder. 

### Examples
Train a base model on Bach chorales.
```
python main.py --train --config=transformer_bach/bach_decoder_config.py
```

Update a base model through augmentive generation. Suppose the base model is at `models/base`. We first need to make a copy of the model, then we can update the copied model.
```
cp -r models/base models/aug_gen
python main.py --load --update --config=models/aug_gen/config.py
```

Generate some output from a trained model.
```
python mainm.py --load --generate --config=models/aug_gen/config.py
```
