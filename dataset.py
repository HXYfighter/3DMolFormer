import torch
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from scipy.spatial.transform import Rotation as R

from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary
from data_utils import normalize_coords, Mol2SmilesCoords, get_mol

class ProtLigDataset(torch.utils.data.Dataset):
    def __init__(self, vol_path, protein_lmdb=None, ligand_lmdb=None, prot_times=1, coords_max=20, coords_min=-20, max_num=0):
        self._tokenizer = ProtLigTokenizer()
        self._vocabulary = read_vocabulary(vol_path)
        self.coords_max = coords_max
        self.coords_min = coords_min
        self.prot_times = prot_times # repeating each protein multiple times

        self.protein_keys = []
        self.ligand_keys = []

        if protein_lmdb is not None:
            self.protein_env = lmdb.open(protein_lmdb, subdir=False, readonly=True, lock=False,
                                         readahead=False, meminit=False, max_readers=256)
            self.protein_txn = self.protein_env.begin()
            self.protein_keys = list(self.protein_txn.cursor().iternext(values=False))
            if max_num > 0:
                self.protein_keys = self.protein_keys[:min(max_num, len(self.protein_keys))]

        if ligand_lmdb is not None:
            self.ligand_env = lmdb.open(ligand_lmdb, subdir=False, readonly=True, lock=False,
                                        readahead=False, meminit=False, max_readers=256)
            self.ligand_txn = self.ligand_env.begin()
            self.ligand_keys = list(self.ligand_txn.cursor().iternext(values=False))
            if max_num > 0:
                self.ligand_keys = self.ligand_keys[:min(max_num, len(self.ligand_keys))]

    def __getitem__(self, i):
        if i < self.prot_times * len(self.protein_keys):
            # Fetch protein data
            protein_index = i // self.prot_times
            datapoint_pickled = self.protein_txn.get(self.protein_keys[protein_index])
            data = pickle.loads(datapoint_pickled)
            prot_coords = normalize_coords(data['coordinates'][0])
            if np.max(prot_coords) > self.coords_max or np.min(prot_coords) < self.coords_min:
                return None
            tokens, numbers = self._tokenizer.tokenize(protein=data['atoms'], prot_coords=prot_coords)
            encoded = self._vocabulary.encode(tokens)
        else:
            # Fetch ligand data
            ligand_index = i - self.prot_times * len(self.protein_keys)
            i_lig = ligand_index // 10
            i_conf = ligand_index % 10
            datapoint_pickled = self.ligand_txn.get(self.ligand_keys[i_lig])
            data = pickle.loads(datapoint_pickled)
            lig_coords = normalize_coords(data['coordinates'][i_conf])
            if np.max(lig_coords) > self.coords_max or np.min(lig_coords) < self.coords_min:
                return None
            tokens, numbers = self._tokenizer.tokenize(ligand_SMILES=data['smi'], ligand_atoms=data['atoms'], lig_coords=lig_coords)
            encoded = self._vocabulary.encode(tokens)

        if encoded[0] == -1 or len(encoded) > 2047:
            return None
        return encoded[:-1], encoded[1:], numbers[:-1], numbers[1:]

    def __len__(self):
        return self.prot_times * len(self.protein_keys) + 10 * len(self.ligand_keys)

    def voc_len(self):
        return self._vocabulary.__len__()


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, vol_path, pair_lmdb=None, max_num=0):
        self._tokenizer = ProtLigTokenizer()
        self._vocabulary = read_vocabulary(vol_path)

        self.pair_keys = []

        if pair_lmdb is not None:
            self.pair_env = lmdb.open(pair_lmdb, subdir=False, readonly=True, lock=False,
                                      readahead=False, meminit=False, max_readers=256)
            self.pair_txn = self.pair_env.begin()
            self.pair_keys = list(self.pair_txn.cursor().iternext(values=False))
            if max_num > 0:
                self.pair_keys = self.pair_keys[:min(max_num, len(self.pair_keys))]

    def __getitem__(self, i):
        datapoint_pickled = self.pair_txn.get(self.pair_keys[i])
        data = pickle.loads(datapoint_pickled)
        prot_coords = data['protein_pos'].numpy() - data['ligand_center_of_mass'].numpy()

        p = np.random.uniform()
        lig_mol = get_mol(data['ligand_element'].tolist(), 
                          data['ligand_bond_index'][0].tolist(), 
                          data['ligand_bond_index'][1].tolist(), 
                          data['ligand_bond_type'].tolist(),
                          data['ligand_pos'].tolist())
        if lig_mol is None:
            return None
        smiles, lig_atoms, lig_coords = Mol2SmilesCoords(lig_mol, canonical=False)
        lig_coords = lig_coords - data['ligand_center_of_mass'].numpy()

        tokens, numbers = self._tokenizer.tokenize(protein=data['protein_atom_name'], prot_coords=prot_coords,
                                                   ligand_SMILES=smiles, ligand_atoms=lig_atoms, lig_coords=lig_coords)
        encoded = self._vocabulary.encode(tokens)
        if encoded[0] == -1 or len(encoded) > 2047:
            return None
        return encoded[:-1], encoded[1:], numbers[:-1], numbers[1:]

    def __len__(self):
        return len(self.pair_keys)

    def voc_len(self):
        return self._vocabulary.__len__()


class ProtLigPairDataset(torch.utils.data.Dataset):
    def __init__(self, vol_path, pair_lmdb=None, max_num=0, rand_aug=0.5, rot_aug=True):
        self._tokenizer = ProtLigTokenizer()
        self._vocabulary = read_vocabulary(vol_path)
        self.rand_aug = rand_aug # SMILES randomization as data augmentation
        self.rot_aug = rot_aug # 3D rotation as data augmentation

        self.pair_keys = []

        if pair_lmdb is not None:
            self.pair_env = lmdb.open(pair_lmdb, subdir=False, readonly=True, lock=False,
                                      readahead=False, meminit=False, max_readers=256)
            self.pair_txn = self.pair_env.begin()
            self.pair_keys = list(self.pair_txn.cursor().iternext(values=False))
            if max_num > 0:
                self.pair_keys = self.pair_keys[:min(max_num, len(self.pair_keys))]

    def __getitem__(self, i):
        datapoint_pickled = self.pair_txn.get(self.pair_keys[i])
        data = pickle.loads(datapoint_pickled)
        center = [data['config']['cx'], data['config']['cy'], data['config']['cz']]
        prot_coords = data['holo_pocket_coordinates'][0] - center
        lig = data['holo_mol']

        p = np.random.uniform()
        smiles, lig_atoms, lig_coords = Mol2SmilesCoords(lig, canonical=(p < self.rand_aug))
        
        lig_coords = lig_coords - center
        if self.rot_aug:
            random_rotation = R.random().as_matrix()
            prot_coords = prot_coords @ random_rotation.T
            lig_coords = lig_coords @ random_rotation.T

        tokens, numbers = self._tokenizer.tokenize(protein=data['pocket_atoms'], prot_coords=prot_coords,
                                                   ligand_SMILES=smiles, ligand_atoms=lig_atoms, lig_coords=lig_coords)
        encoded = self._vocabulary.encode(tokens)
        if encoded[0] == -1 or len(encoded) > 2047:
            return None
        return encoded[:-1], encoded[1:], numbers[:-1], numbers[1:]

    def __len__(self):
        return len(self.pair_keys)

    def voc_len(self):
        return self._vocabulary.__len__()


def collate_fn(encoded_seqs):
    encoded_seqs = [item for item in encoded_seqs if item is not None]
    max_length = max([len(seq[0]) for seq in encoded_seqs])

    collated_arr_x = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padding with zeroes
    collated_arr_y = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)
    collated_arr_x_num = torch.zeros(len(encoded_seqs), max_length, dtype=torch.float)
    collated_arr_y_num = torch.zeros(len(encoded_seqs), max_length, dtype=torch.float)

    for i, seq in enumerate(encoded_seqs):
        collated_arr_x[i, :len(seq[0])] = torch.as_tensor(seq[0], dtype=torch.long)
        collated_arr_y[i, :len(seq[1])] = torch.as_tensor(seq[1], dtype=torch.long)
        collated_arr_x_num[i, :len(seq[2])] = torch.as_tensor(seq[2], dtype=torch.float)
        collated_arr_y_num[i, :len(seq[3])] = torch.as_tensor(seq[3], dtype=torch.float)

    return collated_arr_x, collated_arr_y, collated_arr_x_num, collated_arr_y_num