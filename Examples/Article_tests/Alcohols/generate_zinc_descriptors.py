#!/usr/bin/env python
# -*- coding: utf-8 -*-

#########################################################################################
# filename:	spoc_descriptors_210113_v1.py												#
# author:	YangQi																		#
# date:		2021-01-13																	#
# function： implementation of SPOC descriptors.											#
# licence:	MIT licence																	#
#########################################################################################

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors


def morgan(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 4, nBits=1024)


def spoc_descriptors(name):

    # data loading
    data = pd.read_csv(name+'.csv', encoding='utf-8')
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)

    data['Mol_r_al'] = data['smiles'].apply(Chem.MolFromSmiles)


    # Morgan
    X_r_al = []
    X_r_am = []
    X_p = []
    MACC_al = []
    MACC_am = []
    MACC_p = []
    rdkit_al = []
    rdkit_am = []
    rdkit_p = []
    for i in range(len(data)):
        try:
            r_al = AllChem.GetMorganFingerprintAsBitVect(
            data['Mol_r_al'][i], 4, nBits=1024)
            r_al_MACC = MACCSkeys.GenMACCSKeys(data['Mol_r_al'][i])
            r_al_rdkit = calculator.CalcDescriptors(data['Mol_r_al'][i])
        except:
            print('ERROR Aldehyde Reactant: '+data['smiles'][i])
            continue
        

        X_r_al.append(r_al)
        MACC_al.append(r_al_MACC)
        rdkit_al.append(r_al_rdkit)


    X_r_al = np.array(X_r_al)
    MACC_al = np.array(MACC_al)
    rdkit_al = np.array(rdkit_al)

    # X_r = np.array([AllChem.GetMorganFingerprintAsBitVect(
    #     mol, 4, nBits=1024) for mol in data['Mol_r']])
    # X_p = np.array([AllChem.GetMorganFingerprintAsBitVect(
    #     mol, 4, nBits=1024) for mol in data['Mol_p']])


    # data['fp'] = data["Mol"].apply(morgan)
    # X = data['fp'].values

    # X = MACC.copy()

    X_r_al = np.hstack((MACC_al, X_r_al))

    # X_r_al = MACC_al

    # X_r_al = X_r_al

    # X_r_al = np.hstack((MACC_al, rdkit_al))
    # X_r_am = np.hstack((MACC_am, rdkit_am))
    # X_p = np.hstack((MACC_p, rdkit_p))

    # X_r_al = np.hstack((X_r_al, rdkit_al))
    # X_r_am = np.hstack((X_r_am, rdkit_am))
    # X_p = np.hstack((X_p, rdkit_p))

    # X_r_p = X_r_al - X_p

    # transfer NaN and inf to zero
    where_are_nan = np.isnan(X_r_al)
    where_are_inf = np.isinf(X_r_al)
    X_r_al[where_are_nan] = 0
    X_r_al[where_are_inf] = 0

    df_al = pd.DataFrame(X_r_al, columns=[str(i) for i in range(len(X_r_al[0]))])

    df_al.to_csv('fp_spoc_morgan41024_Maccs_'+name+'_alcohol.csv', index=False)




if __name__ == '__main__':
    name = 'experimental_data_results_of_alcohol_ol'
    spoc_descriptors(name)
