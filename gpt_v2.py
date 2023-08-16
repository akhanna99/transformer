"""
Function of code block: Set up hyperparameters and other configurations for the model

batch_size: Independent sequences being processed in parallel during training
block_size: Max context length for predictions. input is divided into blocks of this size to create training samples
n_embd = Embedding dimension
n_head = Nnumber of attention heads in the multi-head self-attention mechanism
n_layer = Number of transformer blocks in the model
dropout = Used to regularise the model during training
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
import collections
import numpy as np

# hyperparameters
batch_size = 8 #32
block_size = 512 #12 #16
max_iters = 50000
eval_interval = 50
learning_rate = 1e-3 #3e-4 #6e-4
'''
weight_decay = 1e-1
beta1 = 0.9
betas2 = 0.95
grad_clip = 1.0
decay_lr =True
10M parameters 300k tokens
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embd = 768
n_head = 8
n_layer = 8
dropout = 0.1
# ------------
initialTrain=True
retrain=False

with open('alice_in_wonderland.txt', 'r', encoding='utf-8') as f:
    text = f.read()

"""
Function: Word-level tokenisation
"""

words = text.split()
word_counts = {word: count for word, count in collections.Counter(words).items()}
unique_words = sorted(word_counts.keys(), key=lambda word: word_counts[word], reverse=True)
vocab_size = len(unique_words)
stoi = {word: i for i, word in enumerate(unique_words)}
itos = {i: word for i, word in enumerate(unique_words)}
encode = lambda s: [stoi[word] for word in s.split()]
decode = lambda l: ' '.join([itos[i] for i in l])

"""
Function: Data is split into train (90%) and test (10%) sets
"""
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    Function: Generate small batches of data
    ix: random indices between the range of the length of the data and the block size
    Output: inputs (x) and targets (y)
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """
    Function: Estimates the loss of the model on both the training and validation datasets
    model.eval(): Puts the model in evaluation mode. This will cause certain configurations
    such as dropout and batch normalisation to behave differently than during training. This
    ensures the model produces consistent results during evaluation.
    Output: 'out' - Mean loss values for both the train and val datasets
    """
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ Function: This is a representation of one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Function: Performs the self-attention computation
        B - batch size, T - sequence length, C - vocab size
        """

        # 1. Linear projections applied to x to obtain key, query and value vectors
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)

        # 2. Compute attention scores ("affinities") using matrix multiplication
        # scale by C**-0.5 to normalise attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        ## 3. Mask using `tril` to ensure attention only applied to current and previous tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # 4. Normaise attention scores using softmax function
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # 5. Apply dropoiut to attention scores to prevent overfitting during training
        wei = self.dropout(wei)

        # 6. Perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        # 7. 'out' represents the contexualised attention matrix of the attention head
        return out

class MultiHeadAttention(nn.Module):
    """ Function: To run multiple heads of self-attention running in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Function: Combine the attention outputs from all the attention heads into one matrix
        """
        # 1. Concatenate output of all attention head
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        # 2. linear projection transforms the concatenated matrix back to original embedding dimention
        # 3. dropout applied to prevent over-fitting
        out = self.dropout(self.proj(out))

        # 4. projection back into the residual pathway
        return out

class FeedFoward(nn.Module):
    """Function: Simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # linear transformation to create the 'hidden' layer
            nn.Linear(n_embd, 4 * n_embd),
            # ReLU activation function introduces non-linearity to the neural network
            nn.ReLU(),
            # projection layer going back into the residual pathway
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Function: A transformer block that does communication followed by computation.
    Allows infromation to be communicated and processed in parallel through multiple
    attention heads and then combined using residual connections """

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # 'fork off' and do some communication and then come back

        # 1. Apply self-attention mechanism to x and add this to the original input tensor x
        # allows information to flow through Transformer block without getting lost in the layers
        # through residual connections
        x = x + self.sa(self.ln1(x))

        # 2. Updated matrix x is passed through feedforward neural network
        # result is added back to tensor x through residual connections
        x = x + self.ffwd(self.ln2(x))

        # 3. Final output represents the processed tensor after both self-attention
        # and feedforward computation
        return x

class BigramLanguageModel(nn.Module):
    """ Large language model that generates sequences of tokens based on a given input text"""
    def __init__(self):
        super().__init__()
        # converts each token to coressponding token embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # creates another embedding layer to add positional encodings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # stacks multiple transformer blocks / layers
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        # final layer normaisation and linear layer to map output embeddings to logits over vocab size
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Function: Computes the model's predictions (logits) for each token in vocab at each
        position in the generated sequence
        Training: Calculates logits and loss based on targets provided
        Testing: Calculates logits but loss is not calculated
        """
        B, T = idx.shape # idx and targets are both (B,T) tensor of integers

        # 1. Token embedding look-up
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # 2. Positional encoding look-up
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        # 3. Token embedding and positional encodings are combined to form input to transformer blocks
        x = tok_emb + pos_emb # (B,T,C)

        # 4. Inputs passed through stack of transformer blocks
        x = self.blocks(x) # (B,T,C)

        # 5. Output of transformer blocks normalised
        x = self.ln_f(x) # (B,T,C)

        # 6. Embeddings converted to logits over the vocab size
        logits = self.lm_head(x) # (B,T,vocab_size)

        # 7. If in train mode, loss is calculated using cross-entropy loss between predicted logits and targets,
        # otherwise, loss is set to None
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Function: Generates new tokens iteratively, taking context (idx) into account
        idx: (B, T) array of indices in the current context
        max_new_tokens: The number of new tokens to be iteratively generated
        """
        for _ in range(max_new_tokens):
            # 1. Take the last `block_size` tokens from the curreent context `idx_cond`
            idx_cond = idx[:, -block_size:]

            # 2. Use the model to predict the next token probabilities based on current context
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] # becomes (B, C)
            probs = F.softmax(logits, dim=-1) # (B, C)

            # 3. Sample from the distribution to find the index of the next token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # 4. Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        # 5. Output the original context and the generated tokens
        return idx

class EarlyStopper:
    """Function: Implements early stopping during model training.
    Keeps track of the min validation loss and no. consecutive epochs were the val loss hasn't improved
    If the val loss doesn't improve for a specifed no. epochs it will return True to early stop"""

    def __init__(self,patience=1,min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter=0
        self.min_val_loss = np.inf

    def early_stop(self, val_loss, model):
        if val_loss < self.min_val_loss:
            # if val loss has improved, update the min val loss and reset the counter
            self.min_val_loss = val_loss
            self.counter = 0
            # save the trained model with the best val loss
            model_path = "trained_model.pth"
            torch.save(model.state_dict(), model_path)

        # if val loss has not improved, increment the counter
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            # if the counter exceeds patience, return True for early stopping
            if self.counter >= self.patience:
                return True
            return False
        
def train_model(model, optimizer, max_iters, eval_interval):
    """Function: Trains the transformer using an optimiser"""

    early_stopper = EarlyStopper(patience = 5, min_delta = 0.01)
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        if early_stopper.early_stop(losses['train'],model):
            print('Early stopping')
            break

        # 1. Sample a batch of data (inputs and targets)
        xb, yb = get_batch('train')

        # 2. Set the model to 'train' model and evaluate the logits and loss
        model.train()
        logits, loss = model(xb, yb)

        # 3. Zero the gradients of the optimiser to clear accumulated gradients for model parameters
        optimizer.zero_grad(set_to_none=True)

        # 4. Perform backpropogation to compute gradients
        loss.backward()

        # 5. Update model parameters
        optimizer.step()

"""Initial training of the transformer, must be run once once to set-up initial "trained_model.pth" """

if initialTrain:
    model = BigramLanguageModel()
    model = model.to(device)

    # set to training mode
    model.train()

    # optimiser to update the model parameters during training
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_model(model, optimizer, max_iters, eval_interval)

    # save the trained model
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)

"""Loads the currently saved transformer model and re-trains it"""

if retrain:
    model = BigramLanguageModel()

    # load the latest saved model
    model_path = "trained_model.pth"
    model.load_state_dict(torch.load(model_path))

    # set to training mode and train
    model = model.to(device)
    model.train()
    learning_rate = 1e-2
    max_iters = 1500
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_model(model, optimizer, max_iters, eval_interval)

    # save the trained model
    model_path = "trained_model.pth"
    torch.save(model.state_dict(), model_path)

""" Inference code for generating new tokens """

# Load the latest saved trained model
model = BigramLanguageModel()
model_path = "/content/MyDrive/MyDrive/Colab Notebooks/trained_model_joe.pth"
model.load_state_dict(torch.load(model_path, map_location='cpu'))
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Set up a random index as the starting context for the text generation
# context = torch.zeros((1, 1), dtype=torch.long, device=device)

# Specify an initial prompt for the text generation
prompt = "Alice sat on the rabbit"
context2 = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

# Generate and print the new tokens
generated_tokens = model.generate(context2, max_new_tokens=300)[0].tolist()
generated_text = decode(generated_tokens)
print(generated_text)