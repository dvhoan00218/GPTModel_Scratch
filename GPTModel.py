import math
import warnings
import torch
from torch import nn
import torch.nn.functional as F

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)
    
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiheadAttention(config)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(config.ff_dropout),
        )
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
    
class RelativeGlobalMultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.max_len = config.max_len
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        assert self.d_model % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.Er = nn.Parameter(torch.randn(max_len, d_head))
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(max_len, max_len))
            .unsqueeze(0).unsqueeze(0)
        )
        # self.mask.shape = (1, 1, max_len, max_len)

    
    def forward(self, x):
        # x.shape == (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        # k_t.shape = (batch_size, num_heads, d_head, seq_len)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape = (batch_size, num_heads, seq_len, d_head)
        
        start = self.max_len - seq_len
        Er_t = self.Er[start:, :].transpose(0, 1)
        # Er_t.shape = (d_head, seq_len)
        QEr = torch.matmul(q, Er_t)
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        Srel = self.skew(QEr)
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        
        QK_t = torch.matmul(q, k_t)
        # QK_t.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = (QK_t + Srel) / math.sqrt(q.size(-1))
        mask = self.mask[:, :, :seq_len, :seq_len]
        # mask.shape = (1, 1, seq_len, seq_len)
        attn = attn.masked_fill(mask == 0, -1e+6)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        # out.shape = (batch_size, num_heads, seq_len, d_head)
        out = out.transpose(1, 2)
        # out.shape == (batch_size, seq_len, num_heads, d_head)
        out = out.reshape(batch_size, seq_len, -1)
        # out.shape == (batch_size, seq_len, d_model)
        return self.dropout(out)
        
    
    def skew(self, QEr):
        # QEr.shape = (batch_size, num_heads, seq_len, seq_len)
        padded = F.pad(QEr, (1, 0))
        # padded.shape = (batch_size, num_heads, seq_len, 1 + seq_len)
        batch_size, num_heads, num_rows, num_cols = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_cols, num_rows)
        # reshaped.size = (batch_size, num_heads, 1 + seq_len, seq_len)
        Srel = reshaped[:, :, 1:, :]
        # Srel.shape = (batch_size, num_heads, seq_len, seq_len)
        return Srel    
    
class MultiheadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.num_heads = config.num_heads
        assert d_model % self.num_heads == 0, "invalid heads and embedding dimension configuration"
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.proj_dropout = nn.Dropout(config.ff_dropout)
        self.register_buffer(
            "mask", 
            torch.tril(torch.ones(config.max_len, config.max_len))
            .unsqueeze(0).unsqueeze(0)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        # x.shape == (batch_size, seq_len, d_model)
        k_t = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        # shape == (batch_size, num_heads, seq_len, head_dim)
        
        attn = torch.matmul(q, k_t) / math.sqrt(q.size(-1))
        # print(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        mask = self.mask[:, :, :seq_len, :seq_len]
        attn = attn.masked_fill(mask == 0, -1e+6)
        attn = self.attn_dropout(attn)
        # attn.shape == (batch_size, num_heads, seq_len, seq_len)
        attn = F.softmax(attn, dim=-1)
        y = torch.matmul(attn, v)
        # y.shape == (batch_size, num_heads, seq_len, head_dim)
        y = y.transpose(1, 2)
        # y.shape == (batch_size, seq_len, num_heads, head_dim)
        y = y.reshape(batch_size, seq_len, -1)
        # y.shape == (batch_size, seq_len, d_model)
        y = self.proj_dropout(self.proj(y))
        return y
    

class GPTConfig:
    attn_dropout = 0.1
    embed_dropout = 0.1
    ff_dropout = 0.1
    
    def __init__(
        self, vocab_size, max_len, **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_len = max_len
        for key, value in kwargs.items():
            setattr(self, key, value)

class GPT1Config(GPTConfig):
    def __init__(self, vocab_size, max_len, num_head, num_block, d_model):
        super().__init__(vocab_size, max_len)
        self.num_heads = num_head
        self.num_blocks = num_block
        self.d_model = d_model
    
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_model = config.d_model
        self.max_len = config.max_len
        self.num_blocks = config.num_blocks
        self.tok_embed = InputEmbeddings(d_model=d_model,
                                         vocab_size=config.vocab_size
        )
        self.pos_embed = PositionalEncoding(d_model=d_model,
                                            seq_len=config.max_len)
        
        self.dropout = nn.Dropout(config.embed_dropout)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.num_blocks)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, config.vocab_size)
    
    def forward(self, x, target=None):
        # batch_size = x.size(0)
        seq_len = x.size(1)
        assert seq_len <= self.max_len, "sequence longer than model capacity"
        
        x = self.tok_embed(x)
        # tok_embedding.shape == (batch_size, seq_len, d_model)
        x = self.pos_embed(x)
        # pos_embedding.shape == (1, seq_len, d_model)
        # x = self.dropout(tok_embedding + pos_embedding)
        # x = self.dropout(tok_embedding)
        x = self.blocks(x)
        # print(x)
        x = self.ln(x)
        # print(x)
        x = self.fc(x)
        # x.shape == (batch_size, seq_len, vocab_size)
        return x
    
    def generate(self, idx: torch.Tensor, EOS_idx:int , max_new_tokens: int):
        # idx is (B, T) array of indices in the current context
        # print(self.num_blocks)
        # print(idx)
        # print(idx[:, -self.num_blocks:])
        for _ in range(max_new_tokens):
            # crop the context too the  last block_size tokens
            # because tokens don't communicate between blocks
            # idx_crop = idx[:, -self.num_blocks:]
            # get the predictions
            logits = self.forward(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution with probabilities probs
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1) 
            if idx_next == EOS_idx:
                break
            
        return idx
    

def build_GPTModel(vocab_size: int,
                      max_len: int,
                      num_head: int=12,
                      num_block: int=12,
                      d_model: int=768) -> GPT:
    
    # Create the GPT model
    config = GPT1Config(vocab_size, max_len, num_head, num_block, d_model)
    model = GPT(config)
    
    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # config = get_config()

    GPTModel = build_GPTModel(vocab_size=10,
                              max_len=12,
                              num_head=8,
                              num_block=12,
                              d_model=512)
    
    print(GPTModel)