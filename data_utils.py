import numpy as np
from rdkit import Chem
from rdkit.Chem import rdchem
from rdkit.Geometry import Point3D
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.error')
ptable = rdchem.GetPeriodicTable()

def normalize_coords(coordinates):
    return coordinates - coordinates.mean(axis=0)

def Mol2SmilesCoords(mol, canonical=False):
    mol = Chem.RemoveHs(mol)
    conf = mol.GetConformer()
    smiles = Chem.MolToSmiles(mol, canonical=canonical, doRandom=(not canonical))
    mol_smiles = Chem.MolFromSmiles(smiles)
    atom_map = {atom.GetIdx(): mol.GetSubstructMatch(mol_smiles)[atom.GetIdx()] for atom in mol_smiles.GetAtoms()}
    atom_seq = [atom.GetSymbol() for atom in mol_smiles.GetAtoms()]
    coordinates = np.array([conf.GetAtomPosition(atom_map[atom.GetIdx()]) for atom in mol_smiles.GetAtoms()])
    return smiles, atom_seq, coordinates

def get_mol(atom_ids, bond_starts, bond_ends, bond_types, coords):
    mol = Chem.RWMol()
    atoms = []
    for atom_id in atom_ids:
        atom = Chem.Atom(atom_id)
        mol.AddAtom(atom)
        atoms.append(ptable.GetElementSymbol(atom_id))

    for start, end, bond_type in zip(bond_starts, bond_ends, bond_types):
        if start > end:
            continue
        if bond_type == 1:
            mol.AddBond(start, end, rdchem.BondType.SINGLE)
        elif bond_type == 2:
            mol.AddBond(start, end, rdchem.BondType.DOUBLE)
        elif bond_type == 3:
            mol.AddBond(start, end, rdchem.BondType.TRIPLE)
        elif bond_type == 4:
            mol.AddBond(start, end, rdchem.BondType.AROMATIC)

    conformer = Chem.Conformer(len(atom_ids))
    for i in range(len(atom_ids)):
        conformer.SetAtomPosition(i, Point3D(coords[i][0], coords[i][1], coords[i][2]))
    mol.AddConformer(conformer)
    try:
        Chem.SanitizeMol(mol)
        return mol
    except:
        return None

def calc_rmsd(matrix1, matrix2):
    if matrix1.shape != matrix2.shape or matrix1.shape[1] != 3:
        raise ValueError("Both matrices must have the shape (N, 3).")

    diff_squared = np.square(matrix1 - matrix2)
    rmsd = np.sqrt(np.mean(np.sum(diff_squared, axis=1)))
    
    return rmsd