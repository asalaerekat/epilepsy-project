import os, random
import torch, torch.nn as nn, torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
import numpy as np, pandas as pd
import torchvision.transforms as T

from dataset import VideoDataset, VideoAugmentation
from models import get_model
from config import get_args
from utils import accuracy, compute_metrics, plot_loss_curves

def train_epoch(model, loader, crit, opt, device, clip=1.0):
    model.train()
    total_loss=0; all_out=[]; all_lbl=[]
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out,y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total_loss += loss.item()*x.size(0)
        all_out.append(out.detach()); all_lbl.append(y.detach())
    avg_loss = total_loss/len(loader.dataset)
    out = torch.cat(all_out); lbl = torch.cat(all_lbl)
    return avg_loss, accuracy(out,lbl), compute_metrics(out,lbl)

def eval_epoch(model, loader, crit, device):
    model.eval()
    total_loss=0; all_out=[]; all_lbl=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            total_loss += crit(out,y).item()*x.size(0)
            all_out.append(out); all_lbl.append(y)
    avg_loss = total_loss/len(loader.dataset)
    out = torch.cat(all_out); lbl = torch.cat(all_lbl)
    return avg_loss, accuracy(out,lbl), compute_metrics(out,lbl), out

def main():
    args = get_args()
    # seeds & determinism
    seed=42
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic=True; cudnn.benchmark=False

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.plot_dir,   exist_ok=True)

    train_tf = T.Compose([
        T.RandomResizedCrop((112,112), scale=(0.8,1.0), ratio=(0.9,1.1)),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(0.2,0.2,0.2,0.1),
        T.RandomGrayscale(0.1),
        T.GaussianBlur((3,3), sigma=(0.1,2.0)),
        T.ToTensor(),
        T.Normalize([0.43216,0.394666,0.37645],
                    [0.22803,0.22145,0.216989]),
    ])
    val_tf   = T.Compose([
        T.Resize((112,112)),
        T.ToTensor(),
        T.Normalize([0.43216,0.394666,0.37645],
                    [0.22803,0.22145,0.216989]),
    ])

    train_transform = VideoAugmentation(train_tf)
    val_transform   = VideoAugmentation(val_tf)


    full_ds = VideoDataset(args.video_dir,
                           num_frames=args.num_frames,
                           transform=None)
    files = full_ds.video_files

    test_prefixes = {
        '102_1', '105_1', '107_1', '108_1', '109_1', '116_1',
        '119_1', '26_1', '29_1', '37_1', '41_1', '44_1',
        '46_1', '52_1', '66_1', '70_1', '7_4', '81_1',
        '91_1', '93_1', '9_4'
    }
    
    test_ids = [i for i,f in enumerate(files) if any(f.startswith(p) for p in test_prefixes)]
    trainval_ids = [i for i in range(len(files)) if i not in test_ids]

    trainval_ds = Subset(full_ds, trainval_ids)
    test_ds     = Subset(full_ds, test_ids)
    # assign deterministic transforms to test
    test_ds.dataset.transform = val_transform

    # save test manifest
    pd.DataFrame({"filename":[files[i] for i in test_ids]})\
      .to_excel(os.path.join(args.output_dir,"test_split.xlsx"),index=False)

    kf = KFold(n_splits=args.k_folds, shuffle=True, random_state=seed)
    for fold,(tr_idx,cv_idx) in enumerate(kf.split(trainval_ds), start=1):
        print(f"\n=== Fold {fold}/{args.k_folds} ===")
        tr_sub = Subset(trainval_ds, tr_idx)
        cv_sub = Subset(trainval_ds, cv_idx)
        # assign per-split transforms
        tr_sub.dataset.transform = train_transform
        cv_sub.dataset.transform = val_transform

        tr_ld = DataLoader(tr_sub, batch_size=args.batch_size, shuffle=True,  num_workers=1)
        cv_ld = DataLoader(cv_sub, batch_size=args.batch_size, shuffle=False, num_workers=1)

        # model & optimizer (head-only)
        model = get_model(args.model, args.pretrained, args.num_classes).to(device)
        optimizer = optim.Adam(model.fc.parameters(),
                               lr=args.learning_rate,
                               weight_decay=args.weight_decay)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=5,
                                              gamma=0.2)

        best_loss = float('inf')
        train_losses, val_losses = [], []
        for ep in range(1, args.epochs+1):
            tl, ta, _ = train_epoch(model, tr_ld, criterion, optimizer, device)
            vl, va, _, _ = eval_epoch(model, cv_ld, criterion, device)
            train_losses.append(tl); val_losses.append(vl)
            print(f"[F{fold}E{ep}] tr={tl:.3f}/{ta:.2f} | val={vl:.3f}/{va:.2f} | lr={scheduler.get_last_lr()[0]:.2e}")
            scheduler.step()
            if vl < best_loss:
                best_loss = vl
                torch.save(model.state_dict(),
                           os.path.join(args.output_dir, f"best_fold{fold}.pth"))

        plot_loss_curves(train_losses, val_losses, fold, args.plot_dir)

    print("\n=== Final evaluation on test set ===")
    # test uses only val_transform (no randomness)
    test_ds.dataset.transform = val_transform

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=1
    )

    # load the best model from fold `best_fold`
    final_model = get_model(args.model, args.pretrained, args.num_classes).to(device)
    ckpt_path   = os.path.join(args.output_dir, f"best_model_fold{args.best_fold}.pth")
    final_model.load_state_dict(torch.load(ckpt_path, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_metrics, _ = eval_epoch(
        final_model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.3f}, Acc: {test_acc:.3f}")
    print(
        f"Precision: {test_metrics['precision']:.3f} | "
        f"Recall:    {test_metrics['recall']:.3f}    | "
        f"F1:        {test_metrics['f1']:.3f}        | "
        f"AUC:       {test_metrics['auc']:.3f}        | "
        f"Specificity: {test_metrics['specificity']:.3f}"
    )
    
    
if __name__=="__main__":
    main()
