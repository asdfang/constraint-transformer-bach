# Augmentative Generation for Transformer Bach

This repo contains the implementation of the ICML ML4MD Workshop papers [Bach or Mock? A Grading Function for Chorales in the Style of J.S. Bach](https://arxiv.org/abs/2006.13329) and [Incorporating Music Knowledge in Continual Dataset Augmentation for Music Generation](https://arxiv.org/abs/2006.13331). 

For code specifically relating to the grading function, see `Grader/`.

## Setup
A conda environment is provided in `environment.yml`. Load it with `conda env create -f environment.yml`. The environment is named `tbach` and can be activated with `conda activate tbach`.

## Data
To create the folder of Bach chorales in `chorales/bach_chorales`, run
```
python scripts/create_bach_dataset.py
```

## Train models
Every time we run `main.py`, we can do one of three things:
1. load a model with the `--load` flag
2. train a model with the `--aug_gen` flag for the model trained with augmentative generation
3. train a model with the `--base` flag for the base model

To replicate our experiments, first set the desired `'savename'` and `'description'` for the model in `transformer_bach/bach_decoder_config.py`. The given description is saved in `models/model_id/README.txt`. Then, to train the augmented model, run
```
python main.py --aug_gen --config=transformer_bach/bach_decoder_config.py
```
To train the base model, run
```
python main.py --base --config=transformer_bach/bach_decoder_config.py
```

On the first run, the dataset will need to be created in `data/`. Enter `y` and then `index` when prompted to create the most general vocabulary possible. After building the dataset, training should start. Models are saved in the `models/` folder. Generations are saved in the `models/model_id/generations` folder. 

Throughout model training, `dataset_sizes.csv`, `grades.csv`, and `loss.csv` will be populated with useful information. To visualize this data, run
```
python experiments/plot_training.py --model_dir=models/model_id
```
The plots will be created in the `models/model_id/plots` folder.

## Generate
Use the `--generate` flag to load and generate from a model.
```
python main.py --load --config=models/model_id/config.py --generate
```

## References
If you use our code for research, please cite our paper(s)!

```
@inproceedings{fang2020gradingfunction,
    title={Bach or Mock? {A} Grading Function for Chorales in the Style of {J.S. Bach}},
    author={Fang, Alexander and Liu, Alisa and Seetharaman, Prem and Pardo, Bryan},
    booktitle={Machine Learning for Media Discovery (ML4MD) Workshop at the International Conference on Machine Learning (ICML)},
    year={2020}
}

@inproceedings{liu2020auggen,
    title={Incorporating Music Knowledge in Continual Dataset Augmentation for Music Generation},
    author={Liu, Alisa and Fang, Alexander and Hadjeres, Ga{\"e}tan and Seetharaman, Prem and Pardo, Bryan},
    booktitle={Machine Learning for Media Discovery (ML4MD) Workshop at the International Conference on Machine Learning (ICML)},
    year={2020}
}
```