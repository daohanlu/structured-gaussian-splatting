import time
from typing import List, Optional

import torch.nn as nn
import torch
import torch.nn.functional as F


# Positional encoding (NeRF section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires=10, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    # embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embedder_obj, embedder_obj.out_dim


class Decoder(nn.Module):
    def __init__(
            self,
            latent_size: int,
            hidden_sizes: List[int],
            output_dim: int,
            pos_emb_size: int = 0,
            pos_embed_fn: Optional[Embedder] = None,
            dropout: Optional[List[int]] = None,
            dropout_prob: float = 0.0,
            norm_layers: List[int] = [0, 1, 2, 3, 4, 5, 6, 7],
            latent_in=(),
            weight_norm=True,
            pos_emb_in_all=None,
            use_tanh=False,
            latent_dropout=False,
            residual=True,
    ):
        super(Decoder, self).__init__()

        dims = [latent_size + pos_emb_size] + hidden_sizes + [output_dim]

        self.latent_size = latent_size
        self.pos_emb_size = pos_emb_size
        self.output_dim = output_dim
        self.pos_embed_fn = pos_embed_fn
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)
        self.pos_emb_in_all = pos_emb_in_all
        self.weight_norm = weight_norm
        self.residual = residual

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.pos_emb_in_all and layer != self.num_layers - 2:
                    out_dim -= self.pos_emb_size

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    torch.nn.utils.parametrizations.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                    (not weight_norm)
                    and self.norm_layers is not None
                    and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        # self.th = nn.Tanh() # disable deepSDF's output tanh activation -- we don't want an activation here

    """Deep SDF autodecoder modified for structured Gaussians"""

    # input: N x (L+3)
    def _forward(self, input):
        pos_emb = input[:, -self.pos_emb_size:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, pos_emb], 1)
            raise NotImplementedError()
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            residual = x
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.pos_emb_in_all:
                x = torch.cat([x, pos_emb], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                        self.norm_layers is not None
                        and layer in self.norm_layers
                        and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                if self.residual and layer != 0:
                    x += residual
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)
        return x

    def forward(self, latents: torch.Tensor, xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert len(latents.shape) == 2, 'Inputs must have shape (batch, dim)!'
        assert latents.shape[1] == self.latent_size, (f'self.latent_size if {self.latent_size}, but input latents have '
                                                      f'shape {latents.shape[1]}!')
        if xyz is None:
            assert self.pos_emb_size == 0 and self.pos_embed_fn is None
            return self._forward(latents)
        else:
            assert len(xyz.shape) == len(latents.shape), 'Inputs must have shape (batch, dim)!'
            assert xyz.shape[0] == latents.shape[0], 'Batch size must be the same for xyz and latents!'
            assert xyz.shape[1] == 3, 'xyz must be 3-D!'
            pos_embeds = self.pos_embed_fn.embed(xyz.detach())
            return self._forward(torch.cat((pos_embeds, latents), dim=1))


def test():
    N_strctures = 1000000 // 16
    device = torch.device('cuda:0')

    # -------With Pos Embedding-------
    pos_embed_fn, pos_embed_size = get_embedder()
    decoder = Decoder(128, [2048] * 8, 56 * 16, pos_emb_size=pos_embed_size, pos_embed_fn=pos_embed_fn).to(device)
    print(f'Initialized decoder with latent size {decoder.latent_size}, pos_embed_size {decoder.pos_emb_size}')
    latents = torch.randn((N_strctures, decoder.latent_size), device=device)
    xyz = torch.randn((N_strctures, 3), device=device)
    for i in range(5):
        x = decoder(latents, xyz)

    t = time.time()
    num_runs = 10
    for i in range(num_runs):
        x = decoder(latents, xyz)
    print(f'Decoding took {(time.time() - t) * 1000 / num_runs:.1f} ms')

    # -------No Pos Embedding-------
    decoder = Decoder(128, [2048] * 8, 56 * 16).to(device)
    print(f'Initialized decoder with latent size {decoder.latent_size}, pos_embed_size {decoder.pos_emb_size}')
    latents = torch.randn((N_strctures, decoder.latent_size), device=device)
    for i in range(5):
        x = decoder(latents)

    t = time.time()
    num_runs = 10
    for i in range(num_runs):
        x = decoder(latents)
    print(f'Decoding took {(time.time() - t) * 1000 / num_runs:.1f} ms')


if __name__ == '__main__':
    test()
