import torch
import torch.nn as nn
from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.model.gnn import *


class PretrainedDrugEncoder(nn.Module):
    def __init__(self):
        super(PretrainedDrugEncoder, self).__init__()
        self.gnn = load_pretrained('gin_supervised_infomax')
        self.readout = AvgPooling()

    def forward(self, bg):
        node_feats = [bg.ndata.pop('atomic_number'), bg.ndata.pop('chirality_type')]
        edge_feats = [bg.edata.pop('bond_type'), bg.edata.pop('bond_direction_type')]

        node_feats = self.gnn(bg, node_feats, edge_feats)
        graph_feats = self.readout(bg, node_feats)
        return graph_feats


class MaskBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        hidden_activation="ReLU",
        reduction_ratio=2,
        dropout_rate=0.0,
        layer_norm=True,
    ):
        super(MaskBlock, self).__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim * reduction_ratio)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * reduction_ratio), hidden_dim),
        )
        hidden_layers = [nn.Linear(hidden_dim, output_dim, bias=False)]
        if layer_norm:
            hidden_layers.append(nn.LayerNorm(output_dim))
        hidden_layers.append(nn.ReLU())
        if dropout_rate > 0:
            hidden_layers.append(nn.Dropout(p=dropout_rate))
        self.hidden_layer = nn.Sequential(*hidden_layers)

    def forward(self, X, H):
        v_mask = self.mask_layer(X)
        v_out = self.hidden_layer(v_mask * H)
        return v_out


class SerialMaskNet(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_units=[],
        hidden_activations="ReLU",
        reduction_ratio=1,
        dropout_rates=0.1,
        layer_norm=True,
    ):
        super(SerialMaskNet, self).__init__()
        if not isinstance(dropout_rates, list):
            dropout_rates = [dropout_rates] * len(hidden_units)
        if not isinstance(hidden_activations, list):
            hidden_activations = [hidden_activations] * len(hidden_units)
        self.hidden_units = [input_dim] + hidden_units
        self.mask_blocks = nn.ModuleList()
        self.gating = nn.Linear(input_dim, 1, bias=False)
        for idx in range(len(self.hidden_units) - 1):
            self.mask_blocks.append(
                MaskBlock(
                    input_dim,
                    self.hidden_units[idx],
                    self.hidden_units[idx + 1],
                    hidden_activations[idx],
                    reduction_ratio,
                    dropout_rates[idx],
                    layer_norm,
                )
            )

    def forward(self, X):
        v_out = X
        v_outs = []
        gating_scores = []
        for idx in range(len(self.hidden_units) - 1):
            v_out = self.mask_blocks[idx](X, v_out)
            score = self.gating(X)
            v_outs.append(v_out)
            gating_scores.append(score)

        v_outs = torch.stack(v_outs, 2)

        gating_scores = torch.stack(gating_scores, 1)  # (bs, num_experts, 1)
        moe_out = torch.matmul(v_outs, gating_scores.softmax(1))
        v_outs = moe_out.squeeze()  # + X  # (bs, in_features)

        return v_outs


class PointWiseGateBlock(nn.Module):
    def __init__(self, hidden_size, dropout_prob=0.5):
        super(PointWiseGateBlock, self).__init__()
        self.proj = nn.Linear(4 * hidden_size, hidden_size)
        self.Norm1 = nn.LayerNorm(hidden_size)
        self.Norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x, g):
        x = self.dropout1(self.Norm1(x))
        g = self.dropout2(self.Norm2(g))
        m = torch.cat([x, g, torch.mul(x, g), torch.sub(x, g)], dim=-1)
        gate = torch.sigmoid(self.proj(m))
        x = torch.mul(gate, x) + torch.mul((1 - gate), g)
        return x


class graph11_new(nn.Module):
    def __init__(self, in_size, output_dim, hidden=50, dropout=0.5):
        super(graph11_new, self).__init__()

        self.bifusion = PointWiseGateBlock(in_size)

        self.trifusion = nn.Sequential(
            SerialMaskNet(input_dim=in_size, hidden_units=[in_size] * 3),
            nn.ReLU(),
        )

    def forward(self, cell_features, drug1_feature, drug2_feature):
        mafu_c_d1 = self.bifusion(cell_features, drug1_feature)
        mafu_c_d2 = self.bifusion(cell_features, drug2_feature)
        mafu_d1_d2 = self.bifusion(drug1_feature, drug2_feature)
        mafu_output = mafu_c_d1 + mafu_c_d2 + mafu_d1_d2

        tri_fusion_out = self.trifusion(mafu_output)
        return tri_fusion_out


class DeepGraphFusion(nn.Module):
    def __init__(self, cell_dim):
        super(DeepGraphFusion, self).__init__()

        self.drug_encoder = PretrainedDrugEncoder()

        cell_output_dim = 512
        hidden_dim = 1024

        self.cell_line_mlp = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cell_output_dim),
            nn.ReLU(),
        )

        self.d1_mlp = nn.Sequential(
            nn.Linear(300, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, cell_output_dim),
            nn.ReLU(),
        )

        self.fusion_layer = graph11_new(cell_output_dim, cell_output_dim, cell_output_dim)

        self.predict = nn.Sequential(
            nn.Linear(cell_output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, bg1, bg2, cell_features):
        drug1_feature = self.d1_mlp(self.drug_encoder(bg1))
        drug2_feature = self.d1_mlp(self.drug_encoder(bg2))

        cell_features = self.cell_line_mlp(cell_features)

        final_features = self.fusion_layer(cell_features, drug1_feature, drug2_feature)

        predict = self.predict(final_features)

        return predict
