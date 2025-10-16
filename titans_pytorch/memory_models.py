import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Parameter, ParameterList

from einops import rearrange

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = rearrange(gamma, 'b d -> b 1 d')

        return self.ln(x) * (gamma + 1.)

# norm + residual wrapper, as used in original TTT paper
# but could be removed

class ResidualNorm(Module):
    def __init__(
        self,
        dim,
        model: Module
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.model = model

    def forward(self, x):

        out = self.model(x)

        return self.norm(out) + x

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 2.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)

        self.weights = ParameterList([Parameter(torch.randn(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(
        self,
        x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight
        return x

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x

# an MLP modelled after the popular swiglu ff in modern transformers

class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth = 1, # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),
                Parameter(torch.randn(dim_inner, dim)),
            ]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            x, gates = (x @ w1).chunk(2, dim = -1)

            x = x * F.gelu(gates)

            x = x @ w2

            x = x + residual

        return self.norm(x)

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # queries
            Parameter(torch.randn(dim, dim)), # keys
            Parameter(torch.randn(dim, dim)), # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out

def selective_scan_ref(u, delta, A, B, C, D=None, z=None):
    if u.dim() == 3:
        batch_size, seq_len, d_inner = u.shape
    else:
        seq_len = 1
        batch_size, d_inner = u.shape
    dt_rank = delta.shape[-1]
    d_state = A.shape[-1]

    delta_expanded = delta.unsqueeze(2)  # (B, L, 1, dt_rank)

    if delta_expanded.dim() - A.dim() == 2:
        deltaA = torch.exp(delta_expanded * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)
    else:
        deltaA = torch.exp(delta_expanded * A.unsqueeze(1))
    deltaB = delta.unsqueeze(2) * B.unsqueeze(2)  # (B, L, d_inner, dt_rank)

    h = torch.zeros(batch_size, d_inner, d_state, device=u.device, dtype=u.dtype)
    ys = []
    
    for i in range(seq_len):
        h = deltaA[:, i] * h + deltaB[:, i] * u[:, i].unsqueeze(-1)
        y = (h @ C[:, i].unsqueeze(-1)).squeeze(-1)
        ys.append(y)
    
    y = torch.stack(ys, dim=1)
    
    if D is not None:
        y = y + u * D
    if z is not None:
        y = y * F.silu(z)
        
    return y

class MambaBlock(Module):
    def __init__(
        self,
        dim,
        dt_rank="auto",
        d_state=16,
        d_conv=4,
        expansion_factor=2.0
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank if dt_rank != "auto" else max(16, dim // 16)
        
        d_inner = int(dim * expansion_factor)
        self.d_inner = d_inner
        
        self.weights = ParameterList([
            Parameter(torch.randn(dim, d_inner * 2)),      # x_proj (input projection)
            Parameter(torch.randn(d_inner, 1, d_conv)),       # conv1d weight  
            Parameter(torch.randn(d_inner)),             # conv1d bias
            Parameter(torch.randn(d_inner, self.dt_rank)), # dt_proj
            Parameter(torch.randn(d_inner, d_state)),      # A_log 
            Parameter(torch.randn(d_inner, self.dt_rank)), # B_proj
            Parameter(torch.randn(d_inner, self.dt_rank)), # C_proj  
            Parameter(torch.randn(d_inner)),               # D (skip connection)
            Parameter(torch.randn(d_inner, dim)),          # out_proj
            Parameter(torch.randn(self.dt_rank)),          # dt_proj_bias
        ])

        for i, weight in enumerate(self.weights):
            if i == 4:
                A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1))
                self.weights[i].data = -A_log
            elif i == 7:
                nn.init.ones_(weight)
            elif i == 9:
                nn.init.constant_(weight, 1.0)
    
    def chunked_conv1d(self, x_proj_out, conv_weight, conv_bias, chunk_size=256):
        B, L, D = x_proj_out.shape
        _, _, _, kernel_size = conv_weight.shape
        
        outputs = []
        
        for i in range(0, B, chunk_size):
            end_idx = min(i + chunk_size, B)
            chunk_size_actual = end_idx - i
            
            x_conv_chunk = F.conv1d(
                x_proj_out[i:end_idx].transpose(1, 2),
                conv_weight[i:end_idx].reshape(-1, 1, kernel_size),
                bias=conv_bias[i:end_idx].reshape(-1),
                padding=self.d_conv - 1,
                groups=self.d_inner
            )
            
            outputs.append(x_conv_chunk)
        
        x_conv = torch.cat(outputs, dim=0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return x_conv

    def forward(self, x):
        (x_proj, conv_weight, conv_bias, dt_proj, A_log, 
         B_proj, C_proj, D, out_proj, dt_proj_bias) = self.weights
        
        is_need_squeeze = False
        if x.dim() < 3:
            is_need_squeeze = True
            x = x.unsqueeze(-2)

        x_z = x @ x_proj  # (B, L, 2*d_inner)
        x_proj_out, z = x_z.chunk(2, dim=-1)  # Each (B, L, d_inner)

        if conv_weight.dim() == 3:
            x_conv = F.conv1d(
                x_proj_out.transpose(1, 2),
                conv_weight,        # (d_inner, 1, d_conv)
                bias=conv_bias,     # (d_inner,)
                padding=self.d_conv - 1,
                groups=128
            )
        else:
            x_conv = self.chunked_conv1d(x_proj_out, conv_weight, conv_bias, chunk_size=conv_weight.shape[0] // 4)
        
        x_conv = F.silu(x_proj_out)

        x_proj = x_conv @ dt_proj
        
        if dt_proj_bias.dim() == 2:
            dt = F.softplus(x_proj + dt_proj_bias.unsqueeze(1))
        else:
            dt = F.softplus(x_proj + dt_proj_bias)
        B = x_conv @ B_proj
        C = x_conv @ C_proj
        
        A = -A_log.exp()
        
        if D.dim() == 2:
            y = selective_scan_ref(x_conv, dt, A, B, C, D.unsqueeze(1), z)
        else:
            y = selective_scan_ref(x_conv, dt, A, B, C, D, z)
        
        # Output projection
        output = y @ out_proj

        if is_need_squeeze:
            output = output.squeeze()
        return output
    
class HRMMemoryBlock:
    pass