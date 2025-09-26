import torch 
import os
from torch import nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from IndicTransToolkit.processor import IndicProcessor
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from utils import modify_token
from dataset import Bhili_Dataset

special_tokens = [
    "bhil_Deva", 
    "gondi_Deva",
    "mun_Deva"
]


# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = 'cuda'

tokenizer = AutoTokenizer.from_pretrained(
    "indic_indic_dist_320M_mod", 
    trust_remote_code = True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "indic_indic_dist_320M_mod", 
    trust_remote_code = True, 
    use_cache = False,
    attn_implementation="flash_attention_2"
).to(DEVICE)

# freezing parameters
for param in model.parameters() :
    param.requires_grad = False

# unfreezing specific layers 

for param in model.lm_head.parameters() :
    param.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, 
    weight_decay=0.01
    )


ds = Bhili_Dataset(tokenizer)

# Initialize TensorBoard writer
writer = SummaryWriter(log_dir='runs/indic_trans_sft')

for idx in tqdm(range(int(len(ds) * 0.8))) : 
    model.train()

    # get inputs
    x,y = ds[idx] 
    x['labels'] = y['input_ids']
    x.to(DEVICE)

    out = model(**x)

    loss = out.loss
    print(f"Loss at sample {idx} : {loss.item()}")
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    
    # Log to TensorBoard
    writer.add_scalar('Training/Loss', loss.item(), idx)
    
    # Log learning rate
    current_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar('Training/Learning_Rate', current_lr, idx)
    
    # Log every 100 steps
    if idx % 100 == 0:
        print(f"Step {idx}, Loss: {loss.item():.4f}")
        
        # Log gradient norms
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        writer.add_scalar('Training/Gradient_Norm', total_norm, idx)
    
    # Save checkpoints every 500 steps
    if idx % 500 == 0 and idx > 0:
        checkpoint_dir = f"checkpoints/step_{idx}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        print(f"Saving checkpoint at step {idx}...")
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        
        # Save optimizer state
        torch.save({
            'step': idx,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, os.path.join(checkpoint_dir, 'training_state.pt'))
        
        print(f"Checkpoint saved to {checkpoint_dir}")



# Save final checkpoint
final_checkpoint_dir = "checkpoints/final_model"
os.makedirs(final_checkpoint_dir, exist_ok=True)

print("Saving final model...")
model.save_pretrained(final_checkpoint_dir)
# tokenizer.save_pretrained(final_checkpoint_dir)

# Save final training state
torch.save({
    'step': idx,
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
    'training_completed': True
}, os.path.join(final_checkpoint_dir, 'training_state.pt'))

print(f"Final model saved to {final_checkpoint_dir}")

# Close TensorBoard writer
writer.close()

print("Training completed! TensorBoard logs saved to 'runs/indic_trans_sft'")
print("To view logs, run: tensorboard --logdir=runs")
