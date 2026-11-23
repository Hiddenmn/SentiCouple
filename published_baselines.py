"""Run published fusion architectures on the same frozen input features."""

import importlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F
import yaml

from networks import intensity_anchor, sentiment_shift


UPSTREAM = {
    'lnln': '80a052c62b7b3b7a6cb6956846e9d9c121f56eda',
    'tfr-net': '00f06a68a44a6b6fc0043f01954336f30ec3b090',
    'rm-net': '04efd21bd4999e15892af8072ee90ba6fe0e2e8d',
}

ANCHORED_HEADS = ('anchored-shared', 'anchored-independent')


def zero_projection(layer, outputs):
    # Keep the initialization of all other layers and the stochastic training draws paired.
    with torch.random.fork_rng(devices=[]):
        projection = nn.Linear(layer.in_features, outputs)
    nn.init.zeros_(projection.weight)
    nn.init.zeros_(projection.bias)
    return projection


def import_upstream(name, path):
    path = Path(path).resolve(strict=True)
    revision = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                                       text=True).strip()
    if revision != UPSTREAM[name]:
        raise ValueError(f'{name} reference must be at commit {UPSTREAM[name]}.')
    changes = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain', '-uno'],
                                      text=True).strip()
    if changes:
        raise ValueError('Use the pinned upstream source without tracked edits.')
    sys.path.insert(0, str(path))
    return path


class LNLNFeatures(nn.Module):
    """Retain dominant modality correction and hyper-modality learning."""

    def __init__(self, path, corpus, dimensions, lengths, head):
        super().__init__()
        module = importlib.import_module('models.lnln')
        config = yaml.safe_load((path / f'configs/train_{corpus}.yaml').read_text(encoding='utf-8'))
        text, audio, vision = dimensions
        text_len, audio_len, vision_len = lengths
        extractor = config['model']['feature_extractor']
        extractor['input_dims'] = (text, vision, audio)
        extractor['input_length'] = (text_len, vision_len, audio_len)
        config['model']['dmml']['regression']['out_dim'] = 1 if head == 'regression' else 4
        with patch.object(module, 'BertTextEncoder', return_value=nn.Identity()):
            self.model = module.LNLN(config)
        self.auxiliary_weights = {key: config['base'][key] for key in ('alpha', 'beta', 'gamma')}

    def initialize_correction(self, outputs):
        self.model.dmml[3] = zero_projection(self.model.dmml[3], outputs)

    def forward(self, observed, visible, full=None, native=None):
        text, audio, vision = observed
        input_tuple = (vision, audio, text)
        if full is None:
            # The released prediction path accepts an empty complete-input tuple.
            return self.model((None, None, None), input_tuple)['sentiment_preds'], None
        full_text, full_audio, full_vision = full
        output = self.model((full_vision, full_audio, full_text), input_tuple)
        w = output['w']
        discriminator = output['effectiveness_discriminator_out']
        target = torch.arange(len(discriminator), device=discriminator.device) < len(discriminator) // 2
        weights = self.auxiliary_weights
        auxiliary = (weights['alpha'] * F.mse_loss(w, torch.ones_like(w)) +
                     weights['beta'] * F.cross_entropy(discriminator, target.long()) +
                     weights['gamma'] * F.mse_loss(output['rec_feats'], output['complete_feats']))
        return output['sentiment_preds'], auxiliary


class TorchAttention(nn.MultiheadAttention):
    def forward(self, query, key, value, attn_mask=None):
        return super().forward(query, key, value, attn_mask=attn_mask, need_weights=False)


def use_torch_attention(alignment):
    """Preserve the projected attention while omitting unused weight matrices."""
    layers = [layer for layer in alignment.modules() if 'self_attn' in layer._modules]
    with torch.random.fork_rng(devices=[]):
        for layer in layers:
            upstream = layer.self_attn
            attention = TorchAttention(upstream.embed_dim, upstream.num_heads, dropout=upstream.attn_dropout)
            attention.load_state_dict(upstream.state_dict())
            layer.self_attn = attention


class TFRFeatures(nn.Module):
    """Use the public alignment, reconstruction and gated fusion modules."""

    def __init__(self, dimensions, lengths, head, corpus, visual_only):
        super().__init__()
        from models.missingTask.TFR_NET.alignment import Alignment
        from models.missingTask.TFR_NET.fusion import Fusion
        from models.missingTask.TFR_NET.generator import Generator

        self.lengths = tuple(((length + 24) // 25) * 25 for length in lengths)
        # The published channel-attention block groups the sequence dimension into 25 heads.
        args = SimpleNamespace(seq_lens=self.lengths, feature_dims=dimensions, dst_feature_dim_nheads=(30, 6),
                               alignmentModule='crossmodal_attn', fusionModule='c_gate', generatorModule='linear',
                               nlevels=2 if corpus == 'sims' else 3, text_dropout=.2,
                               conv1d_kernel_size_l=1, conv1d_kernel_size_a=5, conv1d_kernel_size_v=3,
                               attn_dropout=.2, attn_dropout_a=.1, attn_dropout_v=0,
                               relu_dropout=0 if corpus == 'sims' else .2, embed_dropout=.1, res_dropout=.2,
                               attn_mask=True, fusion_t_in=90, fusion_a_in=90, fusion_v_in=90,
                               fusion_t_hid=36, fusion_a_hid=20, fusion_v_hid=48, fusion_gru_layers=3,
                               fusion_drop=.2, cls_hidden_dim=128, cls_dropout=.1)
        self.align = Alignment(args)
        use_torch_attention(self.align)
        self.fuse = Fusion(args)
        if head == 'joint':
            self.fuse.Model.classifier.linear_trans_final = nn.Linear(args.cls_hidden_dim, 4)
        streams = ('vision',) if visual_only else ('audio', 'vision')
        self.generators = nn.ModuleDict({stream: Generator(args, modality=stream) for stream in streams})

    def initialize_correction(self, outputs):
        classifier = self.fuse.Model.classifier
        classifier.linear_trans_final = zero_projection(classifier.linear_trans_final, outputs)

    def forward(self, observed, visible, full=None, native=None):
        padded = [F.pad(values, (0, 0, 0, length - values.shape[1]))
                  for values, length in zip(observed, self.lengths)]
        hidden = self.align(*padded)
        text_positions = torch.arange(self.lengths[0], device=observed[0].device)
        last_text = torch.where(visible[0], text_positions[:visible[0].shape[1]], 0).amax(dim=1)
        text_prefix = text_positions <= last_text[:, None]
        # Fixed AV spans carry no information about which masked values originally existed.
        av_prefix = [(torch.arange(length, device=values.device) < values.shape[1]).expand(len(values), -1)
                     for values, length in zip(observed[1:], self.lengths[1:])]
        fusion_inputs = [(features, prefix.float()) for features, prefix in
                         zip(hidden[:3], [text_prefix, *av_prefix])]
        prediction = self.fuse(*fusion_inputs)
        if full is None:
            return prediction, None
        losses = []
        for index, stream in enumerate(('audio', 'vision'), start=1):
            if stream in self.generators:
                target = full[index]
                imputed = self.generators[stream](hidden[3 + index])[:, :target.shape[1]]
                missing = native[index] & ~visible[index]
                error = F.smooth_l1_loss(imputed, target, reduction='none') * missing[..., None]
                losses.append(error.sum() / (missing.sum().clamp_min(1) * target.shape[-1]))
        return prediction, .5 * torch.stack(losses).mean()


class RMNetFeatures(nn.Module):
    """Adapt the published RM-Net core to a common grid without adding observations."""

    def __init__(self, path, dimensions, lengths):
        super().__init__()
        reference = path / 'code/src/p2p3'
        spec = spec_from_file_location('rmnet_reference', reference / 'model.py')
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        self.config = yaml.safe_load((reference / 'config.yaml').read_text(encoding='utf-8'))
        self.config['dimensions'] = list(dimensions)
        self.steps = self.config['max_length']
        if max(lengths[1:]) > self.steps:
            raise ValueError('The shared audiovisual bins must fit on the released RM-Net time grid.')
        self.model = module.RMNet(self.config)

    def embed_grid(self, streams, masks, native_masks):
        features, visibility = [], []
        for index, (values, mask, native) in enumerate(zip(streams, masks, native_masks)):
            batch, length, width = values.shape
            if index == 0:
                # Place transcript tokens at equal subinterval centers in their observed order.
                rank = native.long().cumsum(dim=1) - 1
                slots = ((2 * rank + 1) * self.steps // (2 * native.sum(1, keepdim=True).clamp_min(1)))
                slots = slots.clamp_min(0)
            else:
                centers = (2 * torch.arange(length, device=values.device) + 1) * self.steps // (2 * length)
                slots = centers.expand(batch, -1)
            grid = values.new_zeros((batch, self.steps, width))
            grid.scatter_add_(1, slots[..., None].expand(-1, -1, width), values * native[..., None])
            if index == 0 and length > self.steps:
                # The released grid keeps every contextual token by averaging shared slots.
                counts = torch.zeros((batch, self.steps), dtype=torch.long, device=values.device)
                counts.scatter_add_(1, slots, native.long())
                grid = grid / counts.clamp_min(1)[..., None]
            observed = torch.zeros((batch, self.steps), dtype=torch.long, device=values.device)
            observed.scatter_add_(1, slots, mask.long())
            features.append(grid)
            visibility.append(observed > 0)
        return features, torch.stack(visibility, dim=1)

    def forward(self, observed, visible):
        features, mask = self.embed_grid(observed, visible, visible)
        length = mask.new_full((len(mask),), self.steps, dtype=torch.long)
        output = self.model(features, mask, length)
        score_logit = (output['y'] / 3).clamp(-.999999, .999999).atanh()
        return torch.cat([output['logits'], score_logit[:, None]], dim=1), None

    def training_loss(self, observed, visible, full, native, labels, score, class_weights):
        features, native_mask = self.embed_grid(full, native, native)
        _, visible_mask = self.embed_grid(observed, visible, native)
        length = native_mask.new_full((len(native_mask),), self.steps, dtype=torch.long)
        clean = self.model(features, native_mask, length)
        masked = self.model(features, native_mask, length, visible_mask)
        target = 3 * score
        task = F.l1_loss(clean['y'], target) + F.l1_loss(masked['y'], target)
        present = masked['obs'].any(-1)
        unimodal = ((masked['y_m'] - target[:, None]).abs() * present).sum() / present.sum().clamp_min(1)
        classification = (F.cross_entropy(clean['logits'], labels, weight=class_weights) +
                          F.cross_entropy(masked['logits'], labels, weight=class_weights)) / 2
        reconstruction = []
        for index, (recovered, value) in enumerate(zip(masked['recon'], features)):
            hidden = native_mask[:, index] & ~masked['obs'][:, index]
            error = F.smooth_l1_loss(recovered, value, reduction='none').mean(-1)
            reconstruction.append((error * hidden).sum() / hidden.sum().clamp_min(1))
        consistency = (F.mse_loss(masked['z'], clean['z'].detach()) +
                       F.l1_loss(masked['y'], clean['y'].detach()))
        weights = self.config
        return (task + weights['lambda_unimodal'] * unimodal + weights['lambda_class'] * classification +
                weights['lambda_reconstruction'] * torch.stack(reconstruction).mean() +
                weights['lambda_consistency'] * consistency +
                weights['lambda_interaction'] * masked['y_int'].abs().mean())


class AnchoredReadout(nn.Module):
    """Change only the route from a published fusion network to intensity."""

    def __init__(self, backbone, anchor, class_centers, shared):
        super().__init__()
        self.backbone = backbone
        self.backbone.initialize_correction(3 if shared else 4)
        self.anchor = anchor.requires_grad_(False).eval()
        self.shared = shared
        self.register_buffer('class_centers', class_centers.detach().clone())

    def train(self, mode=True):
        super().train(mode)
        self.anchor.eval()
        return self

    def forward(self, observed, visible, full=None, native=None):
        correction, auxiliary = self.backbone(observed, visible, full, native)
        _, prior = self.anchor(observed[0], *visible)
        present = visible[1].any(dim=1) | visible[2].any(dim=1)
        logits = prior + correction[:, :3] * present[:, None]
        shift = sentiment_shift(logits, prior) if self.shared else correction[:, 3] * present
        score = intensity_anchor(prior, self.class_centers) + shift
        return torch.cat([logits, score[:, None]], dim=1), auxiliary


def build_published(name, upstream, corpus, dimensions, lengths, head, visual_only):
    path = import_upstream(name, upstream)
    template = 'sims' if corpus in ('omg', 'msp-face') else corpus
    if name == 'lnln':
        return LNLNFeatures(path, template, dimensions, lengths, head)
    if name == 'rm-net':
        return RMNetFeatures(path, dimensions, lengths)
    return TFRFeatures(dimensions, lengths, head, template, visual_only)
