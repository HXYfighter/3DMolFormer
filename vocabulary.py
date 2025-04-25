import numpy as np
from tqdm import tqdm

from tokenizer import ProtLigTokenizer

class Vocabulary:
    def __init__(self, tokens=None, starting_id=0):
        self._tokens = {}
        self._current_id = starting_id

        if tokens:
            for token, idx in tokens.items():
                self._add(token, idx)
                self._current_id = max(self._current_id, idx + 1)

    def __getitem__(self, token_or_id):
        return self._tokens[token_or_id]

    def add(self, token):
        """Adds a token."""
        if not isinstance(token, str):
            raise TypeError("Token is not a string")
        if token in self:
            return self[token]
        self._add(token, self._current_id)
        self._current_id += 1
        return self._current_id - 1

    def update(self, tokens):
        """Adds many tokens."""
        return [self.add(token) for token in tokens]

    def __delitem__(self, token_or_id):
        other_val = self._tokens[token_or_id]
        del self._tokens[other_val]
        del self._tokens[token_or_id]

    def __contains__(self, token_or_id):
        return token_or_id in self._tokens

    def __eq__(self, other_vocabulary):
        return self._tokens == other_vocabulary._tokens

    def __len__(self):
        return len(self._tokens) // 2

    def encode(self, tokens):
        """Encodes a list of tokens as vocabulary indexes."""
        vocab_index = np.zeros(len(tokens), dtype=np.float32)
        for i, token in enumerate(tokens):
            if token in self._tokens.keys():
                vocab_index[i] = self._tokens[token]
            else:
                return [-1]
        return vocab_index

    def decode(self, vocab_index):
        tokens = []
        for idx in vocab_index:
            tokens.append(self._tokens[idx])
        return tokens

    def _add(self, token, idx):
        if idx not in self._tokens:
            self._tokens[token] = idx
            self._tokens[idx] = token
        else:
            raise ValueError("IDX already present in vocabulary")

    def tokens(self):
        """Returns the tokens from the vocabulary"""
        return [t for t in self._tokens if isinstance(t, str)]


def read_vocabulary(vol_path):
    tokens = set()
    with open(vol_path) as f:
        lines = f.readlines()
        for line in lines:
            curr_token = line.replace("\n", "").strip().split(" ")[0]
            tokens.update([curr_token])
    
    vocabulary = Vocabulary()
    vocabulary.update(["<PAD>"] + sorted(tokens))
    return vocabulary


def create_ProtLig_vocabulary(protein_list=[], smiles_list=[], voc_path=None, coords_voc=False):
    tokenizer = ProtLigTokenizer()
    tokens = set()

    # protein tokens
    print("Processing protein tokens ...")
    for prot in tqdm(protein_list[:1]):
        tokens.update(tokenizer.tokenize_protein(prot))
        print(tokenizer.tokenize_protein(prot))

    # ligand SMILES tokens
    print("Processing SMILES tokens ...")
    for smi in tqdm(smiles_list):
        tokens.update(tokenizer.tokenize_ligand(smi))

    # coordinate tokens
    if coords_voc:
        print("Processing coordinate tokens ...")
        for integer in range(-20, 21):
            tokens.update([str(integer)])
        for n in range(1000):
            _, dec = "{:.3f}".format(n / 1000).split('.')
            tokens.update(['.' + dec])
        tokens.update(['<PROT_COORDS_START>', '<PROT_COORDS_END>', '<LIG_COORDS_START>', '<LIG_COORDS_END>'])
        tokens.update(['-0'])
    else:
        tokens.update(['<PROT_COORDS_START>', '<PROT_COORDS_END>', '<LIG_COORDS_START>', '<LIG_COORDS_END>'])
        tokens.update(['[x]', '[y]', '[z]'])

    vocabulary = Vocabulary()
    vocabulary.update(sorted(tokens))

    if voc_path:
        tokens = vocabulary.tokens()
        f = open(voc_path, "w")
        for t in tokens:
            f.write(t + '\n')
        f.close()

    return vocabulary