import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class ModelConfig(PretrainedConfig):
    # todo:不熟悉PretrainedConfig以及如何进行参数传递
    model_type = 'Tiny-K'

    def __init__(
            self,
            dim: int = 768,  # 模型维度
            n_layers: int = 12,  # Transformer的层数
            n_heads: int = 16,  # 注意力机制的头数
            n_kv_heads: int = 8,  # 键值头的数量
            vocab_size: int = 6144,  # 词汇表大小
            hidden_dim: int = None,  # 隐藏层维度
            multiple_of: int = 64,
            norm_eps: float = 1e-5,  # 归一化层的eps
            max_seq_len: int = 512,  # 最大序列长度
            dropout: float = 0.0,  # dropout概率
            flash_attn: bool = True,  # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)


# define a RMSNorm function
# after a hidden_state, shape@[bs,seq,dim]
class RMSNorm(nn.Module):
    def __int__(self, dim, epsilon=1e-5):
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        bs, seq, dim = x.shape
        # todo:不熟悉torch对元素平方累加
        temp = torch.sum(x ** 2, dim=1)
        return x * self.gamma / torch.sqrt(temp / seq + self.epsilon)


# get the real & imaginary of a vector
def get_ri(vector):
    return torch.real(vector), torch.imag(vector)


# calculate the corresponding freq to the attention
# q@[bs,seq,n_head,head_dim], k@[bs,seq,n_kv_head,head_dim]
# rotary_pos_emb@[seq,head_dim//2]
def freq_cal(x_q, x_k, base=10000):
    bs, seq, n_head, head_dim = x_q.shape
    _, _, n_kv_head, _ = x_k.shape

    xq_r, xq_i = get_ri(x_q)
    xk_r, xk_i = get_ri(x_k)

    one2end = torch.arange(1, seq)
    one2dim = base ** (-2 * torch.arange(1, head_dim, 2) / head_dim)

    freq_emb = torch.outer(one2end, one2dim)  # @[seq, head_dim//2]
    freq_emb = freq_emb.view(1, seq, 1, -1)  # @[1, seq, 1, head_dim//2] to broadcast with x_q
    cos_freq, sin_freq = torch.cos(freq_emb), torch.sin(freq_emb)

    # todo:不熟悉分块旋转
    xq_rot_r = xq_r * cos_freq - xq_i * sin_freq
    xq_rot_i = xq_r * sin_freq + xq_i * cos_freq
    xk_rot_r = xk_r * cos_freq - xk_i * sin_freq
    xk_rot_i = xk_r * sin_freq + xk_i * cos_freq

    return torch.complex(xq_rot_r, xq_rot_i), torch.complex(xk_rot_r, xk_rot_i)


# self-attention to calculate the attention score of hidden_state
# hidden_state@[bs, seq, dim]
class Attention(nn.Module):
    def __int__(self, args: ModelConfig):
        self.args = args
        self.head_dim = self.args.dim // self.args.n_heads
        self.n_reap = self.args.n_heads // self.args.n_kv_heads

        # todo:w_qkv的表达形式不确，错误的在forward中定义
        self.wq = nn.Linear(self.args.dim, self.args.n_heads * self.head_dim)
        self.wk = nn.Linear(self.args.dim, self.args.n_kv_heads * self.head_dim)
        self.wv = nn.Linear(self.args.dim, self.args.n_kv_heads * self.head_dim)

    def forward(self, hidden_state):
        bs, seq, dim = hidden_state.reshape

        # 1) define qkv
        x_q = self.wq(hidden_state).view(bs, seq, self.args.n_heads, self.head_dim)
        x_k = self.wq(hidden_state).view(bs, seq, self.args.n_kv_heads, self.head_dim)
        x_v = self.wq(hidden_state).view(bs, seq, self.args.n_kv_heads, self.head_dim)

        # 2) apply RoPE
        x_q, x_k = freq_cal(x_q, x_k)

        # 3) repeat kv @[bs, seq, n_head, head_dim]
        x_k, x_v = x_k.repeat_interleave(self.n_reap, 2), x_v.repeat_interleave(self.n_reap, 2)

        # 4) apply flash attention or [manual attention]
        # todo:不清楚对于v而言转化为什么形状
        x_q, x_k, x_v = x_q.transpose(1, 2), x_k.transpose(1, 2), x_v.transpose(1, 2)
        x_k = x_k.transpose(2, 3)

        # 5) calculate attention score@[bs, n_head, seq, seq]
        attn_score = torch.matmul(x_q, x_k) / math.sqrt(dim)

        # 6) apply mask matrix@[seq, seq]
        mask = torch.zeros(seq, seq)
        mask = mask.triu(diagonal=1).fill_(-float('inf'))
        attn_score = attn_score + mask[None, None, ...]

        # 7) apply softmax
        # todo:不清楚softmax和v相乘的先后顺序
        attn_score_softmax = torch.softmax(attn_score, dim=-1)

        # 8) qk*v
        return torch.matmul(attn_score_softmax, x_v).transpose(1, 2)


# todo:不清楚MLP这里的作用以及结构
class Mlp(nn.Module):
    def __int__(self, hidden_dim, args: ModelConfig):
        self.args = args

        if hidden_dim is None:
            hidden_dim = 4 * self.args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = self.args.multiple_of * ((hidden_dim + self.args.multiple_of - 1) // self.args.multiple_of)

        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(self.args.dim, self.hidden_dim)
        self.up = nn.Linear(self.args.dim, self.hidden_dim)
        self.activate = nn.SiLU()
        self.down = nn.Linear(self.hidden_dim, self.dim)
        self.out_down = nn.Dropout(self.args.dropout)

    def forward(self, x):
        return self.out_down(self.activate(self.gate(x)) * self.up(x))


# Llama2 consist of multiple Decoder Layers, which be like:
# hidden_state@[bs, seq, dim]
class Decoder(nn.Module):
    def __int__(self, layer_id, args: ModelConfig):
        self.layer_id = layer_id
        self.args = args
        self.activate = RMSNorm(self.args.dim)
        self.attention = Attention()

    def forward(self, hidden_state):
        hidden_state = hidden_state + self.attention(self.activate(hidden_state))
        output = hidden_state + self.Mlp(self.activate(hidden_state))
        return output


# input_ids@[bs, seq]
class Transformer(nn.Module):
    def __int__(self, decoder_layers=24, args=ModelConfig):
        self.args = args

        # todo:如何使用sequential串联忘记
        self.decoders = nn.Sequential()
        self.decoders.append([Decoder(i) for i in range(decoder_layers)])

        self.embedding = nn.Embedding(self.args.vocab_size, 512)
        self.activate = RMSNorm(self.args.dim)

    def forward(self, input_ids):
        hidden_state = self.embedding(input_ids)
        hidden_state = self.decoders(hidden_state)
        return self.activate(hidden_state)

