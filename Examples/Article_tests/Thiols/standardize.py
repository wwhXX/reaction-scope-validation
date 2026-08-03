import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

def standardize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if Descriptors.MolWt(mol) > 300:
            return None
        mol = Chem.RemoveHs(mol)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        return canonical_smiles
    except:
        return None

def process_file(filename):
    df = pd.read_csv(filename)
    df['standardized_smiles'] = df['smiles'].apply(standardize_smiles)
    df = df[df['standardized_smiles'].notna()]
    df = df.drop_duplicates(subset=['standardized_smiles'])
    df['smiles'] = df['standardized_smiles']
    df = df.drop(columns=['standardized_smiles'])
    df.to_csv(filename, index=False)
    return len(df)

ptr_exp_count = process_file('zrz_exp.csv')
ptr_select_count = process_file('zrz_select.csv')

print(f'ptr_exp.csv: {ptr_exp_count} unique molecules')
print(f'ptr_select.csv: {ptr_select_count} unique molecules')
