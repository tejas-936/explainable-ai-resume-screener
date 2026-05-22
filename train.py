import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import config

def loss_function(logits, target_ids, mu, logvar, beta=None):
    """
    Computes HTVAE Loss = Reconstruction Loss (CrossEntropy) + Beta * KL Divergence
    """
    beta = beta if beta is not None else config.BETA
    

    CE = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=0)
    

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = KLD / mu.size(0)
    
    total_loss = CE + (beta * KLD)
    return total_loss, CE, KLD

def train_htvae(model, dataloader, epochs=None, lr=None, accumulation_steps=None, checkpoint_path=None):
    """
    Full training loop for the HTVAE model.
    Includes mixed precision training (AMP) on GPU, gradient clipping,
    gradient accumulation, and model state checkpointing.
    """
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    accumulation_steps = accumulation_steps or config.ACCUMULATION_STEPS
    checkpoint_path = checkpoint_path or config.CHECKPOINT_PATH

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Training on device: {device}")
    
    if use_cuda and torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
        
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    

    scaler = torch.amp.GradScaler('cuda') if use_cuda else None
    
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            state_dict = checkpoint['model_state_dict']

            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict)
            
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss, total_ce, total_kld = 0, 0, 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sentence_mask = batch['sentence_mask'].to(device)
            
            B, S, W = input_ids.size()
            decoder_target_ids = input_ids.view(B, S * W)
            

            if use_cuda:
                with torch.amp.autocast('cuda'):
                    logits, mu, logvar, z = model(input_ids, attention_mask, sentence_mask, decoder_target_ids)
                    loss, ce, kld = loss_function(logits, decoder_target_ids, mu, logvar)
                    loss = loss / accumulation_steps
            else:
                logits, mu, logvar, z = model(input_ids, attention_mask, sentence_mask, decoder_target_ids)
                loss, ce, kld = loss_function(logits, decoder_target_ids, mu, logvar)
                loss = loss / accumulation_steps
            

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            

            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(dataloader)):
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * accumulation_steps
            total_ce += ce.item()
            total_kld += kld.item()
            
            if batch_idx % 50 == 0:
                current_loss = loss.item() * accumulation_steps
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | Loss: {current_loss:.4f} (CE: {ce.item():.4f}, KLD: {kld.item():.4f})")


        epoch_loss = total_loss / len(dataloader)
        save_state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else {},
            'loss': epoch_loss,
        }
        torch.save(save_state, checkpoint_path)
        print(f"Saved checkpoint: epoch {epoch} | Average loss: {epoch_loss:.4f}")
        
    print("Training run completed successfully.")
