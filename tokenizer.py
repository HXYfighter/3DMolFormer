import re
import numpy as np

# tokenizer for protein(pocket)-ligand pairs
class ProtLigTokenizer:
    def __init__(self):
        self.prot_start = '<PROT_START>'
        self.prot_end = '<PROT_END>'
        self.lig_start = '<LIG_START>'
        self.lig_end = '<LIG_END>'
        
        self.prot_coords_start = '<PROT_COORDS_START>'
        self.prot_coords_end = '<PROT_COORDS_END>'
        self.lig_coords_start = '<LIG_COORDS_START>'
        self.lig_coords_end = '<LIG_COORDS_END>'

        SMILES_pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.lig_regex = re.compile(SMILES_pattern)
        self.exclusive_tokens = None

        self.coord_factor = 5

    def tokenize(self, protein=None, prot_coords=None, ligand_SMILES=None, ligand_atoms=None, lig_coords=None):
        tokens = []
        numbers = []
        if protein:
            prot_tokens, prot_numbers = self.tokenize_protein(protein, prot_coords)
            tokens.extend(prot_tokens)
            numbers.extend(prot_numbers)
        if ligand_SMILES:
            lig_tokens, lig_numbers = self.tokenize_ligand(ligand_SMILES, ligand_atoms, lig_coords)
            tokens.extend(lig_tokens)
            numbers.extend(lig_numbers)
        return tokens, numbers

    def tokenize_protein(self, protein, prot_coords=None):
        tokens = []
        numbers = []
        # tokenize protein atoms, remove Hs
        non_hydrogen_indices = [i for i, atom in enumerate(protein) if not atom.startswith('H')]
        alpha_carbon_indices = [i for i, atom in enumerate(protein) if atom == 'CA']
        filtered_atoms = [protein[i][0] if i not in alpha_carbon_indices else 'CA' for i in non_hydrogen_indices]

        if isinstance(prot_coords, np.ndarray):
            assert len(protein) == prot_coords.shape[0]
            # filtered_coords = prot_coords[alpha_carbon_indices, :] / self.coord_factor
            filtered_coords = prot_coords[non_hydrogen_indices, :] / self.coord_factor
            tokens.extend([self.prot_start] + filtered_atoms + [self.prot_end])
            tokens.extend([self.prot_coords_start] + filtered_coords.shape[0] * ['[x]', '[y]', '[z]']
                          + [self.prot_coords_end])
            numbers.extend((len(filtered_atoms) + 3) * [1])
            for i in range(filtered_coords.shape[0]):
                numbers.extend(filtered_coords[i])
            numbers.append(1)
            return tokens, numbers
        else:
            return [self.prot_start] + filtered_atoms + [self.prot_end]

    def tokenize_ligand(self, ligand_SMILES, ligand_atoms=None, lig_coords=None):
        tokens = []
        numbers = []
        # tokenize SMILES string
        tokens_smi = self.lig_regex.findall(ligand_SMILES)
        if self.exclusive_tokens:
            for i, tok in enumerate(tokens_smi):
                if tok.startswith('[') and tok not in self.exclusive_tokens:
                    tokens_smi[i] = '[UNK]'

        if isinstance(lig_coords, np.ndarray):
            assert len(ligand_atoms) == lig_coords.shape[0]
            # remove Hs
            non_hydrogen_indices = [i for i, atom in enumerate(ligand_atoms) if atom != 'H']
            filtered_coords = lig_coords[non_hydrogen_indices, :] / self.coord_factor
            tokens.extend([self.lig_start] + tokens_smi + [self.lig_end])
            tokens.extend([self.lig_coords_start] + filtered_coords.shape[0] * ['[x]', '[y]', '[z]']
                          + [self.lig_coords_end])
            numbers.extend((len(tokens_smi) + 3) * [1])
            for i in range(filtered_coords.shape[0]):
                numbers.extend(filtered_coords[i])
            numbers.append(1)
            return tokens, numbers
        else:
            return [self.lig_start] + tokens_smi + [self.lig_end]

    def untokenize(self, tokens, numbers=None):
        result = {
            'Prot': [],
            'LigSmiles': '',
            'ProtCoords': [],
            'LigCoords': []
        }
        current_type = None

        for i, token in enumerate(tokens):
            if token == self.prot_start:
                current_type = 'Prot'
                result[current_type] = []
            elif token == self.lig_start:
                current_type = 'LigSmiles'
                result[current_type] = ''
            elif token == self.prot_coords_start:
                current_type = 'ProtCoords'
                result[current_type] = []
            elif token == self.lig_coords_start:
                current_type = 'LigCoords'
                result[current_type] = []
            elif token in [self.prot_end, self.lig_end]:
                current_type = None
            elif token in [self.prot_coords_end, self.lig_coords_end]:
                result[current_type] = self.coord_factor * np.array(result[current_type]).reshape(-1, 3)
                current_type = None
                if token == self.lig_coords_end:
                    break
            else:
                if current_type:
                    if current_type == 'LigSmiles':
                        result[current_type] += token
                    elif current_type == 'Prot':
                        result[current_type].append(token)
                    elif current_type in ['ProtCoords','LigCoords'] and token in ['[x]', '[y]', '[z]']:
                        result[current_type].append(numbers[i])

        return result