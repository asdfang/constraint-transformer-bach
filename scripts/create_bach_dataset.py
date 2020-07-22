"""
Create directory of 351 4-part Bach chorales in chorales/bach_chorales/ in XML from music21
File name's index number may not correspond to Riemenschneider index number for chorale.
"""

import sys
sys.path[0] += '/../'

import music21
from tqdm import tqdm
from transformer_bach.utils import ensure_dir

i = 0
ensure_dir('chorales')
ensure_dir('chorales/bach_chorales')
for chorale in tqdm(music21.corpus.chorales.Iterator(1, 371)):
    if len(chorale.parts) == 4:
        for n in chorale.recurse().getElementsByClass('Note'):
            n.lyric = None # remove lyrics
            n.expressions = [] # remove fermatas
        chorale.write('xml', f'chorales/bach_chorales/{i}.xml')
        i += 1