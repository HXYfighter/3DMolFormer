import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from vocabulary import read_vocabulary
from tokenizer import ProtLigTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab = read_vocabulary("ProtLigVoc.txt")
num_token_id = torch.tensor([vocab.__getitem__('[x]'), 
                             vocab.__getitem__('[y]'), 
                             vocab.__getitem__('[z]')]).to(device)
lig_smiles_start_id = torch.tensor(vocab.__getitem__('<LIG_START>')).to(device)
lig_smiles_end_id = torch.tensor(vocab.__getitem__('<LIG_END>')).to(device)
lig_coord_start_id = torch.tensor(vocab.__getitem__('<LIG_COORDS_START>')).to(device)
lig_coord_end_id = torch.tensor(vocab.__getitem__('<LIG_COORDS_END>')).to(device)
tokenizer = ProtLigTokenizer()

def get_lr(it, total_it, learning_rate, warmup):
    warmup_iters = warmup * total_it
    if it < warmup_iters: # linear warmup        
        lr_mult = it / warmup_iters
    else: # cosine learning rate decay        
        decay_ratio  = (it - warmup_iters) / (total_it - warmup_iters)
        lr_mult = max(0.1, 0.5 * (1.0 + np.cos(np.pi * decay_ratio)))
    return learning_rate * lr_mult

def loss_joint(logits, nums, y, y_num, loss_num_w=1.0):
    num_mask = torch.isin(y, num_token_id)
    loss_lm = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1),
                              ignore_index=0, reduction="mean")
    loss_num = F.mse_loss(nums[num_mask], y_num[num_mask].view(-1, 1),
                          reduction="mean")
    loss = loss_lm + loss_num_w * loss_num

    return loss, loss_lm, loss_num

def loss_docking(nums, y, y_num):
    # Only calculate loss on the ligand coordinates
    N, L = y.shape
    lig_coord_mask = torch.zeros_like(y, dtype=torch.bool)
    for i in range(N):
        start_pos = (y[i] == lig_coord_start_id).nonzero(as_tuple=True)[0].item()
        end_pos = (y[i] == lig_coord_end_id).nonzero(as_tuple=True)[0].item()
        range_mask = torch.zeros(L, dtype=torch.bool).to(device)
        range_mask[start_pos + 1:end_pos] = True
        lig_coord_mask[i] = range_mask & torch.isin(y[i], num_token_id)

    loss = F.mse_loss(nums[lig_coord_mask], y_num[lig_coord_mask].view(-1, 1), 
                      reduction="mean")
    return loss

def likelihood(model, seqs, numbers):
    nll_loss = nn.NLLLoss(reduction="none")
    seqs = seqs.cuda()
    numbers = numbers.cuda()
    logits, _ = model(seqs[:, :-1], numbers[:, :-1])
    log_probs = logits.log_softmax(dim=2)
    return nll_loss(log_probs.transpose(1, 2), seqs[:, 1:]).sum(dim=1)

def predict(model, token_sequences, number_sequences, max_len=2048, temperature=1.0, top_k=10, device='cuda'):
    model.eval()
    token_sequences = token_sequences.to(device)
    number_sequences = number_sequences.to(device)
    
    N, seq_len = token_sequences.size()
    output_tokens = token_sequences.clone()
    output_numbers = number_sequences.clone()

    # from tqdm import tqdm
    for step in range(seq_len, max_len):
        with torch.no_grad():
            logits, num_preds = model(output_tokens, output_numbers)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = top_k_indices.gather(-1, next_token_indices.unsqueeze(-1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)

        next_numbers = num_preds[:, -1, :].squeeze(-1)
        output_numbers = torch.cat([output_numbers, next_numbers.unsqueeze(1)], dim=1)

        # Process tokens to find ligand SMILES and count atoms
        for i in range(N):
            if lig_smiles_end_id in output_tokens[i]:
                start_idx = (output_tokens[i] == lig_smiles_start_id).nonzero(as_tuple=True)[0].item()
                end_idx = (output_tokens[i] == lig_smiles_end_id).nonzero(as_tuple=True)[0].item()
                ligand_smiles_codes = output_tokens[i, start_idx:end_idx + 1]
                ligand_smiles_tokens = vocab.decode(ligand_smiles_codes.cpu().numpy())
                ligand_smiles = tokenizer.untokenize(ligand_smiles_tokens)['LigSmiles']
                    
                try:
                    mol = Chem.MolFromSmiles(ligand_smiles)
                    num_atoms = mol.GetNumAtoms()
                except:
                    num_atoms = 0

                curr_idx = len(output_tokens[i]) - 1
                if curr_idx == end_idx + 1:
                    output_tokens[i, -1] = lig_coord_start_id

                coord_start_idx = end_idx + 2
                if curr_idx >= coord_start_idx and curr_idx < coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = num_token_id[(curr_idx - coord_start_idx) % 3]
                elif curr_idx == coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = lig_coord_end_id
                elif curr_idx > coord_start_idx + 3 * num_atoms:
                    output_tokens[i, -1] = 0

            if output_tokens[i, -1] not in num_token_id:
                output_numbers[i, -1] = 1.0
        
        if (output_tokens[:, -1] == 0).all():
            break

    return output_tokens, output_numbers

def predict_smiles(model, token_sequences, number_sequences, max_len=2048, temperature=1.0, top_k=10, device='cuda'):
    model.eval()
    token_sequences = token_sequences.to(device)
    number_sequences = number_sequences.to(device)
    
    N, seq_len = token_sequences.size()
    output_tokens = token_sequences.clone()
    output_numbers = number_sequences.clone()

    from tqdm import tqdm
    for step in tqdm(range(seq_len, max_len), leave=False):
        with torch.no_grad():
            logits, num_preds = model(output_tokens, output_numbers)

        logits = logits[:, -1, :] / temperature
        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token_indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
            next_tokens = top_k_indices.gather(-1, next_token_indices.unsqueeze(-1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        output_tokens = torch.cat([output_tokens, next_tokens.unsqueeze(1)], dim=1)

        next_numbers = num_preds[:, -1, :].squeeze(-1)
        output_numbers = torch.cat([output_numbers, next_numbers.unsqueeze(1)], dim=1)

        # Process tokens to find ligand SMILES and count atoms
        for i in range(N):
            if lig_smiles_end_id in output_tokens[i]:
                end_idx = (output_tokens[i] == lig_smiles_end_id).nonzero(as_tuple=True)[0][0].item()

                curr_idx = len(output_tokens[i]) - 1
                if curr_idx > end_idx:
                    output_tokens[i, -1] = 0

            if output_tokens[i, -1] == lig_coord_end_id:
                output_tokens[i, -1] = 0

            if output_tokens[i, -1] not in num_token_id:
                output_numbers[i, -1] = 1.0
        
        if (output_tokens[:, -1] == 0).all():
            break

    return output_tokens, output_numbers
