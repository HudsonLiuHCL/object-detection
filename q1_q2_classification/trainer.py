from __future__ import print_function

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
from voc_dataset import VOCDataset


def save_this_epoch(args, epoch):
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
        return True
    if args.save_at_end and (epoch + 1) == args.epochs:
        return True
    return False


def save_model(epoch, model_name, model):
    filename = f'checkpoint-{model_name}-epoch{epoch+1}.pth'
    print("Saving model at", filename)
    torch.save(model.state_dict(), filename)


def train(args, model, optimizer, scheduler=None, model_name='model'):
    writer = SummaryWriter()

    # Dataloaders
    train_loader = utils.get_data_loader(
        'voc', train=True, batch_size=args.batch_size, split='trainval', inp_size=args.inp_size
    )
    test_loader = utils.get_data_loader(
        'voc', train=False, batch_size=args.test_batch_size, split='test', inp_size=args.inp_size
    )

    # Move model to device and set to train mode
    model = model.to(args.device)
    model.train()

    cnt = 0
    best_map = 0.0  # Track best validation mAP

    for epoch in range(args.epochs):
        for batch_idx, (data, target, wgt) in enumerate(train_loader):
            data, target, wgt = data.to(args.device), target.to(args.device), wgt.to(args.device)

            optimizer.zero_grad()
            output = model(data)

            # Binary cross-entropy (manual stable implementation)
            eps = 1e-7
            x = output.clamp(-100, 100)
            loss_per_class = torch.maximum(x, torch.zeros_like(x)) - x * target + torch.log1p(torch.exp(-torch.abs(x)))
            weighted_loss = wgt * loss_per_class
            loss = weighted_loss.sum() / (wgt.sum() + eps)

            loss.backward()

            if cnt % args.log_every == 0:
                writer.add_scalar("Loss/train", loss.item(), cnt)
                print(f"Train Epoch: {epoch} [{cnt} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")

                # Log gradients
                for tag, value in model.named_parameters():
                    if value.grad is not None:
                        writer.add_histogram(tag + "/grad", value.grad.cpu().numpy(), cnt)

            optimizer.step()

            # Validation iteration
            if cnt % args.val_every == 0:
                model.eval()
                ap, map = utils.eval_dataset_map(model, args.device, test_loader)
                print("map: ", map)
                writer.add_scalar("map", map, cnt)
                model.train()

                # Save best performing model
                if map > best_map:
                    best_map = map
                    print(f"New best mAP: {best_map:.4f}, saving model as best_resnet18.pth")
                    torch.save(model.state_dict(), "best_resnet18.pth")

            cnt += 1

        if scheduler is not None:
            scheduler.step()
            writer.add_scalar("learning_rate", scheduler.get_last_lr()[0], cnt)

        # Save periodic checkpoint
        if save_this_epoch(args, epoch):
            save_model(epoch, model_name, model)

    # Final evaluation
    ap, map = utils.eval_dataset_map(model, args.device, test_loader)
    print(f"Final test mAP: {map:.4f}")
    torch.save(model.state_dict(), "best_resnet18.pth")
    print("✅ Model saved to best_resnet18.pth")

    return ap, map
