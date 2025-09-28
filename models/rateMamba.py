import torch
import torch.nn as nn
import numpy as np

from mambular.arch_utils.layer_utils.embedding_layer import EmbeddingLayer
from mambular.arch_utils.mamba_utils.mamba_arch import Mamba
from mambular.arch_utils.mamba_utils.mamba_original import MambaOriginal
from mambular.arch_utils.mlp_utils import MLPhead
from mambular.configs.mambular_config import DefaultMambularConfig
from mambular.base_models.utils import BaseModel



class _TinyDWConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 3, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=kernel_size // 2, groups=d_model)
        self.pw1 = nn.Linear(d_model, expansion * d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Linear(expansion * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                 # x: (B, L, D)
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)             # -> (B, D, L)
        x = self.dw(x)
        x = x.transpose(1, 2)             # -> (B, L, D)
        x = self.pw2(self.act(self.pw1(x)))
        x = self.drop(x)
        return residual + x


class _TinyDWConvMixer(nn.Module):
    def __init__(self, d_model: int, n_layers: int = 1, kernel_size: int = 3, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TinyDWConvBlock(d_model, kernel_size, expansion, dropout) for _ in range(max(1, n_layers))]
        )

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class _LearnedAttentionPool(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d_model))
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                 # x: (B, L, D)
        q = self.q[None, None, :]                                  # (1,1,D)
        scores = torch.einsum('bld,b1d->bl1', self.proj(x), q) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=1)                        # (B,L,1)
        return (x * attn).sum(dim=1)                               # (B,D)


class Mambular(BaseModel):


    _DEFAULTS = {
        "mixer_type": "dwconv",        # 'dwconv' | 'identity' | 'mamba'
        "n_mixer_layers": 1,
        "mixer_kernel_size": 3,
        "mixer_expansion": 2,
        "dropout": 0.1,
        "pooling_method": "mean",      # 'mean' | 'attn' | 'mean+attn'
        "shuffle_embeddings": False,
        "d_model_fallback": 128,
        "mamba_version": "mamba-torch"
    }

    def __init__(
        self,
        feature_information: tuple,       # (cat_feature_info, num_feature_info, embedding_feature_info)
        num_classes: int = 1,
        config: DefaultMambularConfig = None,
        **kwargs,
    ):
        if config is None:
            config = DefaultMambularConfig()
        if not hasattr(config, "d_model") or config.d_model is None:
            config.d_model = self._DEFAULTS["d_model_fallback"]
        for k, v in self._DEFAULTS.items():
            if not hasattr(config, k):
                setattr(config, k, v)

        super().__init__(config=config, **kwargs)
        self.save_hyperparameters(ignore=["feature_information"])
        self.returns_ensemble = False

        # Embedding
        self.embedding_layer = EmbeddingLayer(*feature_information, config=config)

        # Mixer
        mixer_type = self.hparams.mixer_type
        if mixer_type == "mamba":
            self.mixer = Mamba(config) if self.hparams.mamba_version == "mamba-torch" else MambaOriginal(config)
        elif mixer_type == "identity":
            self.mixer = nn.Identity()
        else:
            self.mixer = _TinyDWConvMixer(
                d_model=self.hparams.d_model,
                n_layers=int(self.hparams.n_mixer_layers),
                kernel_size=int(self.hparams.mixer_kernel_size),
                expansion=int(self.hparams.mixer_expansion),
                dropout=float(self.hparams.dropout),
            )

        # Optional shuffle
        self.shuffle_embeddings = bool(self.hparams.shuffle_embeddings)
        if self.shuffle_embeddings:
            seq_len = int(getattr(self.embedding_layer, "seq_len", 0) or 0)
            if seq_len > 0:
                self.register_buffer("perm", torch.randperm(seq_len))
            else:
                self.perm = None

        # Pooling
        self.pooling_method = self.hparams.pooling_method
        if "attn" in self.pooling_method:
            self.attn_pool = _LearnedAttentionPool(self.hparams.d_model)

 
        d = self.hparams.d_model

        self.film_gamma = nn.ParameterDict({
            "non_txbf": nn.Parameter(torch.ones(d)),
            "txbf": nn.Parameter(torch.ones(d)),
        })
        self.film_beta = nn.ParameterDict({
            "non_txbf": nn.Parameter(torch.zeros(d)),
            "txbf": nn.Parameter(torch.zeros(d)),
        })

        self.tabular_head_non = MLPhead(input_dim=d, config=config, output_dim=num_classes)
        self.tabular_head_txbf = MLPhead(input_dim=d, config=config, output_dim=num_classes)


        self.active_domain = "non_txbf"
        self.tabular_head = self.tabular_head_non  

        n_inputs = int(np.sum([len(info) for info in feature_information]))
        self.initialize_pooling_layers(config=config, n_inputs=n_inputs)

 
    def set_mode(self, domain: str):
        domain = domain.lower()
        assert domain in ("non_txbf", "txbf")
        self.active_domain = domain
        self.tabular_head = self.tabular_head_non if domain == "non_txbf" else self.tabular_head_txbf

    def get_mode(self):
        return self.active_domain

    def pool_sequence(self, x):
        if self.pooling_method == "mean":
            return x.mean(dim=1)
        elif self.pooling_method == "attn":
            return self.attn_pool(x)
        elif self.pooling_method == "mean+attn":
            return 0.5 * (x.mean(dim=1) + self.attn_pool(x))
        else:
            try:
                return super().pool_sequence(x)
            except AttributeError:
                return x.mean(dim=1)

    def forward(self, *data):
        x = self.embedding_layer(*data)          # (B, L, D)
        if self.shuffle_embeddings and getattr(self, "perm", None) is not None:
            if self.perm.numel() == x.size(1):
                x = x[:, self.perm, :]

        x = self.mixer(x)                        # (B, L, D)
        x = self.pool_sequence(x)                # (B, D)


        dom = self.active_domain
        x = x * self.film_gamma[dom] + self.film_beta[dom]


        if dom == "txbf":
            preds = self.tabular_head_txbf(x)
        else:
            preds = self.tabular_head_non(x)
        return preds
