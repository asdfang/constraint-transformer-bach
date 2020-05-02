# Augmentative Generation for Transformer Bach

## Usage
A conda environment is provided in `environment.yml`. Load it with `conda env create -f environment.yml`. The environment is named `tbach` and can be activated with `conda activate tbach`.

Every time we run `main.py`, we can do any combination of three things:
1. load a model with the `--load` flag
2. train a model with the `--train` flag
3. update a trained model with the `--update` flag

Furthermore, we can generate from the final model with the `--generate` flag. 

On the first run, the dataset will need to be created in `data/`. After building the dataset, training should start. Models are saved in the `models/` folder. Generations are saved in the `models/model_id/generations` folder. 

## Experimental pipeline
1. Create the vocabulary on all Bach chorales. Make `sequences_size=1` in `bach_decoder_config.py`
```
python main.py --create_dicts --config=transformer_bach/bach_decoder_config.py
```
2. Train a base model on Bach chorales. We will used the pickled `index2dicts` from step 1.
```
python main.py --train --config=transformer_bach/bach_decoder_config.py
```
3. Update a base model through augmentive generation. Suppose the base model is at `models/base`. We first need to make a copy of the model, then we can update the copied model.
```
cp -r models/base models/aug_gen
python main.py --load --update --config=models/aug_gen/config.py
```
4. Let's do experiments!
```
python experiments/visualize_grade_dist.py
```