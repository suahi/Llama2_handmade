import math

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from tqdm import tqdm


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
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x, dim):
        temp = torch.sum(x ** 2, dim=-1, keepdim=True)
        out = x * self.gamma / torch.sqrt(temp / dim + self.epsilon)
        return out

    def forward(self, x):
        bs, seq, dim = x.shape
        # todo:不熟悉torch对元素平方累加
        return self._norm(x.float(), dim).type_as(x)


# GQA, repeat n_rep times n_kv_head to n_head
def repeat_kv(x: torch.Tensor, n_reap: int) -> torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头的数量、每个头的维度大小
    bs, seq, n_kv_heads, head_dim = x.shape

    # 如果重复次数为1，则不需要重复，直接返回原始张量
    if n_reap == 1:
        return x

    # 对张量进行扩展和重塑操作以重复键值对
    return (
        x[:, :, :, None, :]  # 在第四个维度（头的维度前）添加一个新的维度
        .expand(bs, seq, n_kv_heads, n_reap, head_dim)  # 将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs, seq, n_kv_heads * n_reap, head_dim)  # 重新塑形，合并键/值对头的数量和重复次数的维度
    )


# get the real & imaginary of a vector
def get_ri(vector):
    # vector@[bs,seq,n_head/n_kv_head,head_dim]
    real = vector[..., ::2]
    imag = vector[..., 1::2]
    return real, imag


# calculate the corresponding freq to the attention
# q@[bs,seq,n_head,head_dim], k@[bs,seq,n_kv_head,head_dim]
# rotary_pos_emb@[seq,head_dim//2]
def freq_cal(dim: int, seq: int, base: float = 10000.0):
    t = torch.arange(seq)
    assert dim % 2 == 0
    freq = 1.0 / base ** (torch.arange(0, dim, 2).float() / dim)

    freq_emb = torch.outer(t, freq)  # @[seq, head_dim//2]
    freq_emb = freq_emb.view(1, seq, 1, -1)  # @[1, seq, 1, head_dim//2] to broadcast with x_q
    cos_freq, sin_freq = torch.cos(freq_emb), torch.sin(freq_emb)

    return cos_freq, sin_freq


def apply_rope(xq, xk, freq_cos, freq_sin):
    xq_r, xq_i = get_ri(xq.float())
    xk_r, xk_i = get_ri(xk.float())

    xq_rot_r = xq_r * freq_cos - xq_i * freq_sin
    xq_rot_i = xq_r * freq_sin + xq_i * freq_cos
    xk_rot_r = xk_r * freq_cos - xk_i * freq_sin
    xk_rot_i = xk_r * freq_sin + xk_i * freq_cos

    # todo:不确定是否穿插
    xq_rot = torch.stack([xq_rot_r, xq_rot_i], dim=-1).flatten(-2)
    xk_rot = torch.stack([xk_rot_r, xk_rot_i], dim=-1).flatten(-2)

    return xq_rot.type_as(xq), xk_rot.type_as(xq)


# self-attention to calculate the attention score of hidden_state
# hidden_state@[bs, seq, dim]
class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()
        self.args = args
        self.head_dim = self.args.dim // self.args.n_heads
        self.n_reap = self.args.n_heads // self.args.n_kv_heads

        # todo:w_qkv的表达形式不确，错误的在forward中定义
        self.wq = nn.Linear(self.args.dim, self.args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.args.dim, self.args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(self.args.dim, self.args.n_kv_heads * self.head_dim, bias=False)

        # 输出权重矩阵。
        self.wo = nn.Linear(self.args.n_heads * self.head_dim, self.args.dim, bias=False)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(self.args.dropout)
        self.resid_dropout = nn.Dropout(self.args.dropout)
        # 保存dropout概率。
        self.dropout = self.args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, self.args.max_seq_len, self.args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲
            self.register_buffer("mask", mask)

    def forward(self, hidden_state, freq_cos, freq_sin):
        bs, seq, dim = hidden_state.shape

        # 1) define qkv
        x_q = self.wq(hidden_state).view(bs, seq, self.args.n_heads, self.head_dim)
        x_k = self.wk(hidden_state).view(bs, seq, self.args.n_kv_heads, self.head_dim)
        x_v = self.wv(hidden_state).view(bs, seq, self.args.n_kv_heads, self.head_dim)

        # 2) apply RoPE
        x_q, x_k = apply_rope(x_q, x_k, freq_cos, freq_sin)

        # 3) repeat kv @[bs, seq, n_head, head_dim]
        x_k, x_v = repeat_kv(x_k, self.n_reap), repeat_kv(x_v, self.n_reap)

        # 4) apply flash attention or [manual attention]
        # todo:不清楚对于v而言转化为什么形状
        x_q, x_k, x_v = x_q.transpose(1, 2), x_k.transpose(1, 2), x_v.transpose(1, 2)
        x_k = x_k.transpose(2, 3)

        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(x_q, x_k, x_v, attn_mask=None,
                                                                      dropout_p=self.dropout if self.training else 0.0,
                                                                      is_causal=True)
        else:
            # 5) calculate attention score@[bs, n_head, seq, seq]
            attn_score = torch.matmul(x_q, x_k) / math.sqrt(self.head_dim)

            # 6) apply mask matrix@[seq, seq]
            attn_score = attn_score + self.mask[..., :seq, :seq]

            # 7) apply softmax
            # todo:不清楚softmax和v相乘的先后顺序
            attn_score_softmax = torch.softmax(attn_score.float(), dim=-1).type_as(x_q)
            attn_score_softmax = self.attn_dropout(attn_score_softmax)

            # 8) qk*v
            output = torch.matmul(attn_score_softmax, x_v).transpose(1, 2).contiguous().view_as(hidden_state)

        output = self.wo(output)
        output = self.resid_dropout(output)

        return output


# todo:不清楚MLP这里的作用以及结构
class Mlp(nn.Module):
    def __init__(self, hidden_dim, args: ModelConfig):
        super().__init__()
        self.args = args

        if hidden_dim is None:
            hidden_dim = 4 * self.args.dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = self.args.multiple_of * ((hidden_dim + self.args.multiple_of - 1) // self.args.multiple_of)

        self.hidden_dim = hidden_dim
        self.gate = nn.Linear(self.args.dim, self.hidden_dim, bias=False)
        self.up = nn.Linear(self.args.dim, self.hidden_dim, bias=False)
        self.activate = nn.SiLU()
        self.down = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.out_down = nn.Dropout(self.args.dropout)

    def forward(self, x):
        return self.out_down(self.activate(self.gate(x)) * self.up(x))


# Llama2 consist of multiple Decoder Layers, which be like:
# hidden_state@[bs, seq, dim]
class Decoder(nn.Module):
    def __init__(self, layer_id, args: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.args = args
        self.hidden_dim = self.args.hidden_dim

        self.attention = Attention(args)
        self.MLP = Mlp(self.hidden_dim, args)

        self.attention_norm = RMSNorm(args.dim)
        self.ffn_norm = RMSNorm(args.dim)

    def forward(self, hidden_state, freq_cos, freq_sin):
        hidden_state = hidden_state + self.attention(self.attention_norm(hidden_state), freq_cos, freq_sin)
        output = hidden_state + self.Mlp(self.ffn_norm(hidden_state))
        return output


# input_ids@[bs, seq]
class Transformer(PreTrainedModel):
    def __int__(self, args: ModelConfig, decoder_layers=24):
        self.args = args

        # todo:如何使用sequential串联忘记
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Decoder(layer_id, args))

        self.embedding = nn.Embedding(self.args.vocab_size, self.args.dim)
        self.llamaRMS = RMSNorm(self.args.dim)
        self.output = nn.Linear(self.args.dim, self.args.vocab_size, bias=False)

        freq_cos, freq_sin = freq_cal(self.args.dim, self.args.max_seq_len)
        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

        self.dropout = nn.Dropout(self.args.dropout)
        self.std = math.sqrt(1.0 / self.args.dim)

        self.last_loss = None
        self.OUT = CausalLMOutputWithPast()  # 输出容器

        self.apply(self._init)
        # 对残差投影进行特殊的缩放初始化
        for pn, p in self.named_parameters():
            if pn.endswith('out_down.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=self.std / math.sqrt(2 * self.args.n_layers))

    def _init(self, module):

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.std)

    def forward(self, input_ids, targets, **kwargs):

        if 'input_ids' in kwargs:
            input_ids = kwargs["input_ids"]
        if 'targets' in kwargs:
            targets = kwargs["targets"]

        # 1) input_ids@[bs, seq] -> hs@[bs, seq, dim]
        hidden_state = self.embedding(input_ids)

        # 2) decoder -> NormLayer -> pred@[bs, seq, dim]
        # 通过Decoder层
        for layer in self.layers:
            hidden_state = layer(hidden_state, self.freqs_cos, self.freqs_sin)

        pred_state = self.llamaRMS(hidden_state)

        # 3) casual: input 0:99 -> output 1:100
        # pred_state@[bs, seq, dim] -> logits@[bs, seq, vocab_size]
        if targets is not None:
            logits = self.output(pred_state)
            self.last_loss = nn.CrossEntropyLoss(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduction='none')
        else:
            logits = self.output[:, -1, :]
            self.last_loss = None

        # 设置输出
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, temperature, top_k, stop_id):
        # 1) input@[bs, seq]
        bs, seq = idx.shape

        for i, _ in tqdm(range(self.args.max_seq_len)):

            # 2) model output@[bs, 1, vocab_size]
            idx = idx if seq <= self.args.max_seq_len else idx[:, -self.args.max_seq_len:]
            logits = self(idx).logits.view(bs, -1)

            # 3) argmax or temperature
            if temperature == 0.0:
                pred = torch.argmax(logits, dim=-1)
            else:
                logits_temperature = logits/ temperature
                logits_k, _ = logits_temperature.topk(k=top_k, dim=-1)
                logits_temperature[logits_temperature < logits_k[:, [-1]]] = -float('Inf')
                logits = torch.softmax(logits_temperature, dim=-1)
                pred = torch.multinomial(logits, num_samples=1)

            idx = torch.concat((idx, pred), dim=1)

            if pred == stop_id:
                break

        return idx[:, seq:]


if __name__ == '__main__':
    x = torch.randn(1, 50)
    model = Transformer(ModelConfig)
    output = model.generate(x, temperature=0.8, top_k=3, stop_id=None)
    print(output.shape)
