import torch
import math

device = "cpu"
class SelfAttention(torch.nn.Module):
    def __init__(self, dmodel, dk, dv, mask, dropout_p=0.2, bias=False):
        super(SelfAttention, self).__init__()
        self.dk = dk
        self.query = torch.nn.Linear(dmodel, dk, bias)
        self.key = torch.nn.Linear(dmodel, dk, bias)
        self.value = torch.nn.Linear(dmodel, dv, bias)
        self.mask = mask
        self.scale = 1/(math.sqrt(dk))
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v):
        # x input (n_batches, Seq_Len, EMB_size)
        # Query (n_batches, Seq_Len, dk)
        queries = self.query(q)
        # keys (n_batches, Seq_Len, dk)
        keys = self.key(k)
        # values (n_batches, Seq_Len, dv)
        values = self.value(v)
        # Matmul(Q,K^T) -> (N_batches, Seq_Len, Seq_len)
        dot = torch.matmul(queries, keys.transpose(-2,-1))
        # Scaling : dividing each by sqrt(dk)
        scaled_dot = dot * self.scale
        # TODO implement the mask
        if self.mask:
            mask = torch.tril(input = torch.ones_like(scaled_dot))
            scaled_dot = scaled_dot.masked_fill(mask == 0, float('-inf'))
        # Running it through a softmax
        # Scaled (N_batches, Seq_Len, Seq_len)
        scaled_dot_attention = torch.nn.functional.softmax(scaled_dot, dim = -1)
        scaled_dot_attention = self.dropout(scaled_dot_attention)
        output = torch.matmul(scaled_dot_attention, values)
        # Output (N_batches, Seq_Len, dv)
        return output
      
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dmodel, dk, dv, dropout_p=0.2, h=6, mask=False, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.heads = torch.nn.ModuleList(

                [SelfAttention(dmodel, dk, dv, mask, dropout_p) for _ in range(h)]

            )
        self.ln = torch.nn.Linear(h*dv, dmodel, bias)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, q, k, v):  # input (N_Batches, Seq_Len, EMB_size)
        concatinated_heads = []
        for _, head in enumerate(self.heads):
            curr_head_out = head(q, k, v)
            concatinated_heads.append(curr_head_out)
        # Concat = (N_Batches, Seq_Len, h * EMB_size)
        output = self.ln(torch.concat(concatinated_heads, -1))
        # output = (N_Batches, Seq_Len, EMB_size)
        return self.dropout(output)
    
    
class PositionWise_FeedForward(torch.nn.Module):

    def __init__(self, dmodel, dropout_p=0.2, hidden_size=2048, bias=False):
        super(PositionWise_FeedForward, self).__init__()
        self.ln1 = torch.nn.Linear(dmodel, hidden_size, bias)
        self.ln2 = torch.nn.Linear(hidden_size, dmodel, bias)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x): # input : (N_Batches, Seq_Len, EMB_size)
        l1_o = self.ln1(x)
        relu_o = torch.nn.functional.gelu(l1_o)
        l2_o = self.ln2(relu_o)
        return self.dropout(l2_o)
    
    
def positional_encoding(n_batch, seq_len, dmodel):
    pe = torch.zeros(n_batch, seq_len, dmodel).to(device)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dmodel, 2).float() * (-torch.log(torch.tensor(10000.0)) / dmodel))
    pe[:, :, 0::2] = torch.sin(position * div_term)
    pe[:, :, 1::2] = torch.cos(position * div_term)
    return pe

class Embeder(torch.nn.Module):
    def __init__(self, dmodel, vocab_size):
        super(Embeder, self).__init__()
        self.emb = torch.nn.Embedding(vocab_size, dmodel)
        self.dmodel = dmodel
    def forward(self, x): # X: (N_Batches, Seq_Len)
        n_batch, seq_len = x.shape
        emb = self.emb(x)
        return emb + positional_encoding(n_batch, seq_len, self.dmodel) # O: (N_Batches, Seq_Len, EMB_size)
    
class DecoderLayer(torch.nn.Module):
    def __init__(self, dmodel, dk, dv, h, dropout_p=0.2, hidden_size=2048):
        super(DecoderLayer, self).__init__()
        self.masked_mh_a = MultiHeadAttention(dmodel, dk, dv, dropout_p, h, True)
        self.layer_norm1= torch.nn.LayerNorm(dmodel)
        self.layer_norm2= torch.nn.LayerNorm(dmodel)
        self.ff= PositionWise_FeedForward(dmodel, dropout_p, hidden_size=hidden_size)

    def forward(self, x, enc_o):
        # X : (N_Batches, Seq_Len, EMB_size), enc_o : (N_Batches, Seq_Len, EMB_size)
        # Apply Layer Norm and masked multi-head attention with residual connection
        x_norm = self.layer_norm1(x)
        masked_mha_o = self.masked_mh_a(x_norm, x_norm, x_norm)
        masked_mha_o = x + masked_mha_o  # Residual connection
        
        # Apply final Layer Norm and feed-forward network with residual connection
        ff_input = self.layer_norm2(masked_mha_o)
        ff_o = self.ff(ff_input)
        ff_o = masked_mha_o + ff_o  # Residual connection
        return ff_o


class Decoder(torch.nn.Module):
    def __init__(self, dmodel, dk, dv, h, vocab_size, dropout_p=0.2, hidden_size = 2048, N=6, bias=False):
        super(Decoder, self).__init__()
        self.dec_layers = torch.nn.ModuleList(
            [DecoderLayer(dmodel, dk, dv, h, dropout_p, hidden_size) for _ in range(N)]

        )
        self.ln = torch.nn.Linear(dmodel, vocab_size, bias)
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x, enc_o):    # X : (N_Batches, Seq_Len, EMB_size), enc_o : (N_Batches, Seq_Len, EMB_size)
        # Calling the forward function for each layer
        for layer in self.dec_layers:
            x = layer(x, enc_o)
        # X :  (N_Batches, Seq_Len, EMB_size)
        return self.dropout(self.ln(x)) # (N_Batches, Seq_Len, Vocabsize)

class Transformer(torch.nn.Module):
    def __init__(self, dmodel, vocab_size, device, h=6, dropout_p=0.2, hidden_size=2048, N=6):
        super(Transformer, self).__init__()
        # Setting the device
        self.device = device
        # Defining the Embedder of the Output language
        self.embedder = Embeder(dmodel, vocab_size)
        # Defining the Decoder
        self.decoder = Decoder(dmodel, dmodel, dmodel, h, vocab_size, dropout_p, hidden_size, N)

    def forward(self, x):
        # [N Batches, Seq Len]
        x = x.to(self.device)
        # [N Batches, Seq Len, EMB size]
        emb = self.embedder(x)  
        x = self.decoder(emb, emb)
        return x

    