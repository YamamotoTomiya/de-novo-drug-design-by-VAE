import ast

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers

import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import BondType
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Draw

RDLogger.DisableLog("rdApp.*")

csv_path = keras.utils.get_file(
    "/Users/tomi/Library/Mobile Documents/com~apple~CloudDocs/NAIST/MI_Lab/research_student/Molecule_Gene_VAE/code/content/250k_rndm_zinc_drugs_clean_3.csv",
    "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv",
)

df = pd.read_csv("content/250k_rndm_zinc_drugs_clean_3.csv")
df["smiles"] = df["smiles"].apply(lambda s: s.replace("\n", ""))
df.head()
smile_list=[]
for i in range(50):
    smile_list.append(df.iloc[i,0])
print(smile_list)

# 化合物のラベルを作成
label_list = ['sample_{}'.format(i) for i in range(len(smile_list))]

# molオブジェクトのリストを作成
mols_list = [Chem.MolFromSmiles(smile) for smile in smile_list]
img = Draw.MolsToGridImage(mols_list,
                           molsPerRow=5, #一列に配置する分子の数
                           subImgSize=(200,200),
                           legends=label_list #化合物の下に表示するラベル
                           )
img.show()