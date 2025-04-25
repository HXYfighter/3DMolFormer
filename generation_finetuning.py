import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import argparse
import glob
import lmdb
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from model import MolFormer
from tokenizer import ProtLigTokenizer
from vocabulary import read_vocabulary
from training_utils import predict, predict_smiles, likelihood
from scoring_function import get_scores, int_div

def read_lmdb(lmdb_path):
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))

    return txn, keys

def memory_update(memory, smiles, scores, seqs, nums):
    seqs_list = [seqs[i, :].cpu().numpy() for i in range(len(smiles))]
    nums_list = [nums[i, :].cpu().numpy() for i in range(len(smiles))]
    main_scores = [score[0] for score in scores]
    docking_scores = [score[1] for score in scores]
    qed_scores = [score[2] for score in scores]
    sa_scores = [score[3] for score in scores]

    for i in range(len(smiles)):
        new_data = pd.DataFrame({"smiles": smiles[i], "scores": main_scores[i], 
                                "docking_scores": docking_scores[i], "qed_scores": qed_scores[i], "sa_scores": sa_scores[i],
                                "seqs": [seqs_list[i]], "numbers": [nums_list[i]]})
        memory = pd.concat([memory, new_data], ignore_index=True, sort=False)

    memory = memory.drop_duplicates(subset=["smiles"])
    memory = memory.sort_values('scores', ascending=False)
    memory = memory.reset_index(drop=True)
    if len(memory) > args.memory_size:
        memory = memory.head(args.memory_size)

    if args.replay > 0:
        s = min(len(memory), args.replay)
        experience = memory.head(5 * args.replay).sample(s)
        experience = experience.reset_index(drop=True)
        smiles += list(experience["smiles"])
        main_scores += list(experience["scores"])
        for index in experience.index:
            seqs = torch.cat((seqs, torch.tensor(experience.loc[index, "seqs"], dtype=torch.long).view(1, -1).cuda()), dim=0)
            nums = torch.cat((nums, torch.tensor(experience.loc[index, "numbers"], dtype=torch.float).view(1, -1).cuda()), dim=0)

    return memory, smiles, main_scores, seqs, nums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--data_path', type=str, default="your_path")
    parser.add_argument('--vocab_path', type=str, default="ProtLigVoc.txt")
    parser.add_argument('--ckpt_load_path', type=str, default="final.pt")
    parser.add_argument('--mol_save_path', type=str, default="generated_mols/")
    # model
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=12, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768, help="embedding dimension", required=False)
    # training
    parser.add_argument('--n_steps', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--sigma', type=float, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--memory_size', type=int, default=1000)
    parser.add_argument('--replay', type=int, default=0)
    args = parser.parse_args()

    writer = SummaryWriter("log_generation/" + args.run_name)
    # if not os.path.exists(args.ckpt_save_path + args.run_name):
        # os.makedirs(args.ckpt_save_path + args.run_name)
    writer.add_text("configs", str(args))

    tokenizer = ProtLigTokenizer()
    vocabulary = read_vocabulary(args.vocab_path)
    prior = MolFormer(vocab_size=vocabulary.__len__(), 
                      d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                      dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
    prior.load_state_dict(torch.load(args.ckpt_load_path), strict=True)
    for param in prior.parameters():
        param.requires_grad = False
    prior.eval()

    txn, keys = read_lmdb(args.data_path + "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")
    ids = torch.load(args.data_path + "crossdocked_pocket10_pose_split.pt")['test']

    for i in range(0, 100):
        agent = MolFormer(vocab_size=vocabulary.__len__(), 
                          d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                          dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
        agent.load_state_dict(torch.load(args.ckpt_load_path), strict=True)
        optimizer = optim.AdamW(agent.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
        agent.eval()

        datapoint_pickled = txn.get(keys[ids[i]])
        data = pickle.loads(datapoint_pickled)
        prot_dir = args.data_path + 'test_set/' + data['protein_filename'].split('/')[0]
        prot_file = glob.glob(prot_dir + "/*.pdb")[0]
        data_tag = str(i) + '-' + data['protein_filename'].split('_rec')[0].replace('/', '-')
        print("Datapoint:", data_tag)
        base_name = os.path.basename(prot_file)
        if not os.path.exists(args.mol_save_path + data_tag):
            os.makedirs(args.mol_save_path + data_tag)

        # prot_coords = data['protein_pos'].numpy() - data['ligand_center_of_mass'].numpy()
        # tokens, numbers = tokenizer.tokenize(protein=data['protein_atom_name'], prot_coords=prot_coords)
        # tokens.append('<LIG_START>')
        # numbers.append(1)
        tokens = ['<LIG_START>']
        numbers = [1]
        encoded = vocabulary.encode(tokens)

        encoded = torch.tensor(encoded, dtype=torch.long)
        encoded_batch = encoded.unsqueeze(0).expand(args.batch_size, -1).to("cuda")
        numbers = torch.tensor(numbers, dtype=torch.float)
        numbers_batch = numbers.unsqueeze(0).expand(args.batch_size, -1).to("cuda")

        memory = pd.DataFrame(columns=["smiles", "scores", "docking_scores", "qed_scores", "sa_scores", "seqs", "numbers"])

        for step in tqdm(range(args.n_steps)):

            predicted_encoded, predicted_numbers = predict_smiles(agent, encoded_batch, numbers_batch, max_len=256, top_k=10)
            smiles = []
            for j in range(args.batch_size):
                predicted_encoded_j = predicted_encoded[j].tolist()
                predicted_tokens_j = vocabulary.decode(predicted_encoded_j)
                predicted_numbers_j = predicted_numbers[j].tolist()
                sample_j = tokenizer.untokenize(predicted_tokens_j, predicted_numbers_j)
                smiles.append(sample_j['LigSmiles'])
            scores = get_scores(smiles, receptor_file=prot_file, box_center=data['ligand_center_of_mass'].numpy())
            
            main_scores = [score[0] for score in scores]
            writer.add_scalar('Mean score ' + data_tag, np.mean(main_scores), step)

            memory, smiles, main_scores, predicted_encoded, predicted_numbers = memory_update(memory, smiles, scores, predicted_encoded, predicted_numbers)

            prior_likelihood = likelihood(prior, predicted_encoded, predicted_numbers)
            agent_likelihood = likelihood(agent, predicted_encoded, predicted_numbers)
            main_scores = torch.tensor(np.array(main_scores)).cuda()
            loss = torch.pow(args.sigma * main_scores - (prior_likelihood - agent_likelihood), 2)
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar('top-100 ' + data_tag, np.mean(np.array(memory['scores'][:100])), step)
            if step % 50 == 0:
                memory.to_csv(args.mol_save_path + args.run_name + data_tag + f"/step{step}.csv")
            if memory.loc[100, 'scores'] > 0.75 or (step >= 200 and memory.loc[100, 'scores'] > 0.70):
                break
            torch.cuda.empty_cache()


        memory.to_csv(args.mol_save_path + args.run_name + data_tag + f"/final.csv")
