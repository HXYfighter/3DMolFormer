import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset import ProtLigDataset, PairDataset, collate_fn
from model import MolFormer
from vocabulary import read_vocabulary
from training_utils import get_lr, loss_joint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--dataset_path', type=str, default="your_path")
    parser.add_argument('--vocab_path', type=str, default="ProtLigVoc.txt")
    parser.add_argument('--ckpt_load_path', type=str, default=None)
    parser.add_argument('--ckpt_save_path', type=str, default="ckpt/")
    # model
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=12, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768, help="embedding dimension", required=False)
    # training
    parser.add_argument('--prot_batch_size', type=int, default=128)
    parser.add_argument('--lig_batch_size', type=int, default=256)
    parser.add_argument('--prot_epochs', type=int, default=2)
    parser.add_argument('--lig_epochs', type=int, default=1)
    parser.add_argument('--pair_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=4e-4)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--warmup', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=40)
    parser.add_argument('--lig_valid_freq', type=int, default=20000)
    parser.add_argument('--prot_valid_freq', type=int, default=20000)
    
    parser.add_argument('--loss_num_w', type=float, default=1.0)

    args = parser.parse_args()

    writer = SummaryWriter("log/" + args.run_name)
    if not os.path.exists(args.ckpt_save_path + args.run_name):
        os.makedirs(args.ckpt_save_path + args.run_name)
    writer.add_text("configs", str(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    # vocab = read_vocabulary(args.vocab_path)
    lig_train_dataset = ProtLigDataset(args.vocab_path, ligand_lmdb=args.dataset_path + "ligands/train.lmdb")
    prot_train_dataset = ProtLigDataset(args.vocab_path, protein_lmdb=args.dataset_path + "pockets/train.lmdb")    
    lig_valid_dataset = ProtLigDataset(args.vocab_path, ligand_lmdb=args.dataset_path + "ligands/valid.lmdb", max_num=10000)
    prot_valid_dataset = ProtLigDataset(args.vocab_path, protein_lmdb=args.dataset_path + "pockets/valid.lmdb", max_num=10000)
    pair_train_dataset = PairDataset(args.vocab_path, pair_lmdb=args.dataset_path + "crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb")


    lig_train_loader = DataLoader(lig_train_dataset, batch_size=args.lig_batch_size, shuffle=True, 
                                  num_workers=32, pin_memory=True, collate_fn=collate_fn)
    prot_train_loader = DataLoader(prot_train_dataset, batch_size=args.prot_batch_size, shuffle=True, 
                                   num_workers=32, pin_memory=True, collate_fn=collate_fn)
    pair_train_loader = DataLoader(pair_train_dataset, batch_size=args.prot_batch_size, shuffle=True, 
                                   num_workers=32, pin_memory=True, collate_fn=collate_fn)
    lig_valid_loader = DataLoader(lig_valid_dataset, batch_size=args.lig_batch_size, shuffle=False, 
                                  num_workers=32, pin_memory=True, collate_fn=collate_fn)
    prot_valid_loader = DataLoader(prot_valid_dataset, batch_size=args.prot_batch_size, shuffle=False, 
                                   num_workers=32, pin_memory=True, collate_fn=collate_fn)

    # Load model
    model = MolFormer(vocab_size=lig_train_dataset.voc_len(), 
                      d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                      dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model: %.2fM" % (n_params / 1e6))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, 
                            betas=(0.9, 0.95))
    if args.ckpt_load_path:
        model.load_state_dict(torch.load(args.ckpt_load_path), strict=True)

    scaler = torch.amp.GradScaler('cuda')
    model = torch.nn.DataParallel(model, device_ids=[0, 1])

    print("Training on Protein Data...")
    num_batches = len(prot_train_loader)
    for epoch in range(args.prot_epochs):
        pbar = tqdm(enumerate(prot_train_loader), total=num_batches)
        for iter_num, (x, y, x_num, y_num) in pbar:
            step = iter_num + num_batches * epoch
            model.train()
            x = x.to(device)
            y = y.to(device)
            x_num = x_num.to(device)
            y_num = y_num.to(device)
            
            with torch.amp.autocast('cuda'):
                with torch.set_grad_enabled(True):
                    logits, nums = model(x, x_num)
                    loss, loss_lm, loss_num = loss_joint(logits, nums, y, y_num, args.loss_num_w)
                    scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                if args.lr_decay:
                    lr = get_lr(step, num_batches * args.prot_epochs, args.learning_rate, args.warmup)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    writer.add_scalar('learning rate', lr, step)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.set_description(f"Step {step}: train loss {loss.item():.5f}, lr {lr:e}")
                writer.add_scalar('protein training loss', loss, step)
                writer.add_scalar('protein training loss_lm', loss_lm, step)
                writer.add_scalar('protein training loss_num', loss_num, step)
            
            # Validation
            if step % args.prot_valid_freq == 0:
                model.eval()
                val_losses = []
                val_losses_lm = []
                val_losses_num = []
                with torch.no_grad():
                    for iter_num, (x, y, x_num, y_num) in tqdm(enumerate(prot_valid_loader), total=len(prot_valid_loader), 
                                                                desc="Protein validation", leave=False):
                        x = x.to(device)
                        y = y.to(device)
                        x_num = x_num.to(device)
                        y_num = y_num.to(device)

                        logits, nums = model(x, x_num)
                        loss, loss_lm, loss_num = loss_joint(logits, nums, y, y_num, args.loss_num_w)

                        val_losses.append(loss.item())
                        val_losses_lm.append(loss_lm.item())
                        val_losses_num.append(loss_num.item())

                val_loss = float(np.mean(val_losses))
                val_losses_lm = float(np.mean(val_losses_lm))
                val_losses_num = float(np.mean(val_losses_num))
                writer.add_scalar('protein validation loss', val_loss, step)
                writer.add_scalar('protein validation loss_lm', val_losses_lm, step)
                writer.add_scalar('protein validation loss_num', val_losses_num, step)

    print("Training on Ligand Data...")
    num_batches = len(lig_train_loader)
    for epoch in range(args.lig_epochs):
        pbar = tqdm(enumerate(lig_train_loader), total=num_batches)
        for iter_num, (x, y, x_num, y_num) in pbar:
            step = iter_num + num_batches * epoch
            model.train()
            x = x.to(device)
            y = y.to(device)
            x_num = x_num.to(device)
            y_num = y_num.to(device)
            
            with torch.amp.autocast('cuda'):
                with torch.set_grad_enabled(True):
                    logits, nums = model(x, x_num)
                    loss, loss_lm, loss_num = loss_joint(logits, nums, y, y_num, args.loss_num_w)
                    scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                if args.lr_decay:
                    lr = get_lr(step, num_batches * args.lig_epochs, args.learning_rate, args.warmup)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    writer.add_scalar('learning rate', lr, step)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.set_description(f"Step {step}: train loss {loss.item():.5f}, lr {lr:e}")
                writer.add_scalar('ligand training loss', loss, step)
                writer.add_scalar('ligand training loss_lm', loss_lm, step)
                writer.add_scalar('ligand training loss_num', loss_num, step)

            # Validation
            if step % args.lig_valid_freq == 0:
                model.eval()
                val_losses = []
                val_losses_lm = []
                val_losses_num = []
                with torch.no_grad():
                    for iter_num, (x, y, x_num, y_num) in tqdm(enumerate(lig_valid_loader), total=len(lig_valid_loader), 
                                                                desc="Ligand validation", leave=False):
                        x = x.to(device)
                        y = y.to(device)
                        x_num = x_num.to(device)
                        y_num = y_num.to(device)
                        
                        logits, nums = model(x, x_num)
                        loss, loss_lm, loss_num = loss_joint(logits, nums, y, y_num, args.loss_num_w)

                        val_losses.append(loss.item())
                        val_losses_lm.append(loss_lm.item())
                        val_losses_num.append(loss_num.item())
                
                val_loss = float(np.mean(val_losses))
                val_losses_lm = float(np.mean(val_losses_lm))
                val_losses_num = float(np.mean(val_losses_num))
                writer.add_scalar('ligand validation loss', val_loss, step)
                writer.add_scalar('ligand validation loss_lm', val_losses_lm, step)
                writer.add_scalar('ligand validation loss_num', val_losses_num, step)

                torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/" + f"step{step}.pt")
                torch.cuda.empty_cache()

    print("Training on Pair Data...")
    num_batches = len(pair_train_loader)
    for epoch in range(args.pair_epochs):
        pbar = tqdm(enumerate(pair_train_loader), total=num_batches)
        for iter_num, (x, y, x_num, y_num) in pbar:
            step = iter_num + num_batches * epoch
            model.train()
            x = x.to(device)
            y = y.to(device)
            x_num = x_num.to(device)
            y_num = y_num.to(device)
            
            with torch.amp.autocast('cuda'):
                with torch.set_grad_enabled(True):
                    logits, nums = model(x, x_num)
                    loss, loss_lm, loss_num = loss_joint(logits, nums, y, y_num, args.loss_num_w)
                    scaled_loss = loss / args.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            if step % args.gradient_accumulation_steps == 0:
                if args.lr_decay:
                    lr = get_lr(step, num_batches * args.pair_epochs, args.learning_rate, args.warmup)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    writer.add_scalar('learning rate', lr, step)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                pbar.set_description(f"Step {step}: train loss {loss.item():.5f}, lr {lr:e}")
                writer.add_scalar('pair training loss', loss, step)
                writer.add_scalar('pair training loss_lm', loss_lm, step)
                writer.add_scalar('pair training loss_num', loss_num, step)
                
    torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/final.pt")