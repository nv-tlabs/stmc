import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding, TimestepEmbedder
from einops import repeat


class TransformerDenoiser(nn.Module):
    name = "transformer"

    def __init__(
        self,
        nfeats: int,
        tx_dim: int,
        latent_dim: int = 512,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
        nb_registers: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()

        self.nfeats = nfeats
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.nb_registers = nb_registers
        self.tx_dim = tx_dim

        # Linear layer for the condition
        self.tx_embedding = nn.Sequential(
            nn.Linear(tx_dim, 2 * latent_dim),
            nn.GELU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Linear layer for the skeletons
        self.skel_embedding = nn.Linear(nfeats, latent_dim)

        # register for aggregating info
        if nb_registers > 0:
            self.registers = nn.Parameter(torch.randn(nb_registers, latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(
            latent_dim, dropout, batch_first=True
        )

        # MLP for the timesteps
        self.timestep_encoder = TimestepEmbedder(latent_dim, self.sequence_pos_encoding)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            norm_first=True,
            activation=activation,
            batch_first=True,
        )

        self.seqTransEncoder = nn.TransformerEncoder(
            seq_trans_encoder_layer, num_layers=num_layers
        )

        # Final layer to go back to skeletons
        self.to_skel_layer = nn.Linear(latent_dim, nfeats)

    def forward(self, x, y, t):
        device = x.device
        x_mask = y["mask"]
        bs, nframes, nfeats = x.shape

        # Time embedding
        time_emb = self.timestep_encoder(t)
        time_mask = torch.ones(bs, dtype=bool, device=device)

        # put all the additionnal here
        info_emb = time_emb[:, None]
        info_mask = time_mask[:, None]

        assert "tx" in y

        # Condition part (can be text/action etc)
        tx_x = y["tx"]["x"]
        tx_mask = y["tx"]["mask"]

        tx_emb = self.tx_embedding(tx_x)

        info_emb = torch.cat((info_emb, tx_emb), 1)
        info_mask = torch.cat((info_mask, tx_mask), 1)

        # add registers
        if self.nb_registers > 0:
            registers = repeat(self.registers, "nbtoken dim -> bs nbtoken dim", bs=bs)
            registers_mask = torch.ones(
                (bs, self.nb_registers), dtype=bool, device=device
            )
            # add the register
            info_emb = torch.cat((info_emb, registers), 1)
            info_mask = torch.cat((info_mask, registers_mask), 1)

        x = self.skel_embedding(x)
        number_of_info = info_emb.shape[1]

        # adding the embedding token for all sequences
        xseq = torch.cat((info_emb, x), 1)

        # add positional encoding to all the tokens
        xseq = self.sequence_pos_encoding(xseq)

        # create a bigger mask, to allow attend to time and condition as well
        aug_mask = torch.cat((info_mask, x_mask), 1)

        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        # extract the important part
        output = self.to_skel_layer(final[:, number_of_info:])
        return output
