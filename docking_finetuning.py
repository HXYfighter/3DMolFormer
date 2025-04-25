import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
import numpy as np
import argparse
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from dataset import ProtLigPairDataset, collate_fn
from model import MolFormer
from vocabulary import read_vocabulary
from training_utils import get_lr, loss_docking

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # base
    parser.add_argument('--run_name', type=str, default="")
    parser.add_argument('--dataset_path', type=str, default="your_path")
    parser.add_argument('--vocab_path', type=str, default="ProtLigVoc.txt")
    parser.add_argument('--ckpt_load_path', type=str, default="final.pt")
    parser.add_argument('--ckpt_save_path', type=str, default="ckpt_docking/")
    # model
    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--n_layer', type=int, default=12, help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=12, help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=768, help="embedding dimension", required=False)
    # training
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--max_epochs', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--lr_decay', type=bool, default=True)
    parser.add_argument('--warmup', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0)
    parser.add_argument('--rand_aug', type=float, default=0.2)
    parser.add_argument('--valid_freq', type=int, default=5)
    
    args = parser.parse_args()

    writer = SummaryWriter("log_docking/" + args.run_name)
    if not os.path.exists(args.ckpt_save_path + args.run_name):
        os.makedirs(args.ckpt_save_path + args.run_name)
    writer.add_text("configs", str(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    vocab = read_vocabulary(args.vocab_path)
    num_token_id = torch.tensor([vocab.__getitem__('[x]'), vocab.__getitem__('[y]'), vocab.__getitem__('[z]')]).to(device)
    train_dataset = ProtLigPairDataset(args.vocab_path, pair_lmdb=args.dataset_path + "test.lmdb",
                                       rand_aug=args.rand_aug, rot_aug=True)
    valid_dataset = ProtLigPairDataset(args.vocab_path, pair_lmdb=args.dataset_path + "valid.lmdb", 
                                       rand_aug=0, rot_aug=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=16, pin_memory=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, 
                              num_workers=16, pin_memory=True, collate_fn=collate_fn)

    # Load model
    model = MolFormer(vocab_size=train_dataset.voc_len(), 
                      d_model=args.n_embd, nhead=args.n_head, num_layers=args.n_layer, 
                      dim_feedforward=4 * args.n_embd, context_length=args.max_length).to("cuda")
    n_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters in the model: %.2fM" % (n_params / 1e6))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                            betas=(0.9, 0.95))
    if args.ckpt_load_path:
        model.load_state_dict(torch.load(args.ckpt_load_path), strict=True)

    scaler = torch.amp.GradScaler('cuda')
    model = torch.nn.DataParallel(model, device_ids=[0,1,2,3])

    num_batches = len(train_loader)
    for epoch in range(args.max_epochs):
        # Training
        pbar = tqdm(enumerate(train_loader), total=num_batches, leave=False)
        for iter_num, (x, y, x_num, y_num) in pbar:
            step = iter_num + num_batches * epoch
            model.train()
            x = x.to(device)
            y = y.to(device)
            x_num = x_num.to(device)
            y_num = y_num.to(device)
            if args.lr_decay:
                lr = get_lr(step, num_batches * args.max_epochs, args.learning_rate, args.warmup)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                writer.add_scalar('learning rate', lr, step)

            with torch.amp.autocast('cuda'):
                with torch.set_grad_enabled(True):
                    logits, nums = model(x, x_num)
                    loss = loss_docking(nums, y, y_num)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            scaler.step(optimizer)
            scaler.update()

            pbar.set_description(f"Epoch {epoch}: train loss {loss.item():.5f}, lr {lr:e}")
            writer.add_scalar('training loss', loss, step)

        # Validation
        if epoch % args.valid_freq == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for iter_num, (x, y, x_num, y_num) in tqdm(enumerate(valid_loader), total=len(valid_loader), 
                                                           desc="Docking validation", leave=False):
                    x = x.to(device)
                    y = y.to(device)
                    x_num = x_num.to(device)
                    y_num = y_num.to(device)

                    logits, nums = model(x, x_num)
                    loss = loss_docking(nums, y, y_num)

                    val_losses.append(loss.item())

            val_loss = float(np.mean(val_losses))
            writer.add_scalar('validation loss', val_loss, epoch)

        if epoch % (10 * args.valid_freq) == 0:
            torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/" + f"epoch{epoch}.pt")
            torch.cuda.empty_cache()

    torch.save(model.module.state_dict(), args.ckpt_save_path + args.run_name + "/final.pt")