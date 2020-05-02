import sys
sys.path[0] += '/../'

import csv
import pandas as pd
import matplotlib.pyplot as plt
from transformer_bach.utils import ensure_dir

gen_folder = 'models/overfit_4-24'

with open(f'{gen_folder}/training_loss.csv', 'r') as fin:
    df = pd.read_csv(fin)
    train_loss = df['train_loss']
    val_loss = df['val_loss']

print(df[df['val_loss'] == df['val_loss'].min()])

plt.figure()
plt.plot(train_loss)
plt.plot(val_loss)
plt.legend(['Training loss', 'Validation loss'])
plt.title('Training curves')
plt.xlabel('Epoch')
plt.ylabel('Loss')
ensure_dir(f'{gen_folder}/plots')
plt.savefig(f'{gen_folder}/plots/training_curves.png')