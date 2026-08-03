from rdkit import Chem

def check_nucleophilic_atoms(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        if any(atom.GetAtomicNum() == 15 for atom in mol.GetAtoms()):
            return 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 16:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        return 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                if atom.GetIsAromatic():
                    if len(atom.GetNeighbors()) == 3:
                        continue
                    else:
                        return 0
                if atom.GetFormalCharge() == 0 and len(atom.GetNeighbors()) == 3:
                    return 0

        return 1
    except Exception as e:
        print(f"Error: {e}")
        return None

def check_tertiary_alcohols(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if mol is None:
            raise ValueError("Invalid SMILES string")

        carbon = None
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 8:
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:
                        for neighbor2 in atom.GetNeighbors():
                            if neighbor2.GetAtomicNum() == 6:
                                if neighbor2.GetIsAromatic():
                                    continue
                                carbon = neighbor2
        if carbon is not None:
            H_count = 0
            for neighbor in carbon.GetNeighbors():
                if neighbor.GetAtomicNum() == 1:
                    H_count += 1
            if H_count == 0:
                return 0
            return 1
        else:
            raise ValueError("No alcohol found")
    except Exception as e:
        print(f"Error: {e}")
        if str(e) == "No alcohol found":
            return 0
        return None


if __name__ == "__main__":
    print(check_tertiary_alcohols("CSC1=C(O)C=CN1"))
