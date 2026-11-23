"""Adapt the released API and UCMR-Net models to matched cached features."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from zipfile import ZipFile

import torch
from torch import nn
from torch.nn import functional as F
import yaml


MOSEI_BASELINES = ('api', 'ucmr-net')
RATING_TO_MOSEI_SCALE = {'mosei': 1., 'mosi': 1., 'sims': 3., 'omg': 3.}


def masked_mean(features, visible):
    return (features * visible.unsqueeze(-1)).sum(1) / visible.sum(1, keepdim=True).clamp_min(1)


class APIForVisualGaps(nn.Module):
    """Reuse API's class prototypes, FiLM and fusion on cached text and vision."""

    def __init__(self, upstream, dimensions, lengths, rating_scale):
        super().__init__()
        from trains.singleTask.model.april import APRIL

        config = json.loads((upstream / 'config/config_april.json').read_text(encoding='utf8'))['april']
        args = SimpleNamespace(**(config['commonParams'] | config['datasetParams']['mosei'] | {
            'use_bert': False, 'use_finetune': False, 'dataset_name': 'mosei',
            'feature_dims': dimensions,
        }))
        self.core = APRIL(args)
        self.rating_scale = rating_scale
        self.contrastive_weight = args.proto_contrastive_weight
        self.reconstruction_weight = args.personal_recon_weight
        for stream, length, layer in (('text', lengths[0], self.core.proj_l),
                                      ('vision', lengths[2], self.core.proj_v)):
            prototype = getattr(self.core, f'proto_{stream}')
            setattr(self.core, f'proto_{stream}', prototype.new_zeros(
                prototype.shape[0], length - layer.kernel_size[0] + 1, prototype.shape[-1]))
        # The comparison uses text and vision. Retain the fusion dimensions of
        # the active branches and omit the absent audio computations.
        for name, module in self.core.named_children():
            if name in ('proj_a', 'trans_a_mem', 'weight_a') or name.startswith(
                ('trans_a_with_', 'trans_l_with_a', 'trans_v_with_a', 'a2', 't2a_', 'v2a_')
            ) or name.endswith('_step'):
                module.requires_grad_(False)

    @staticmethod
    def project(values, visible, layer):
        kernel = layer.kernel_size[0]
        centers = visible.unfold(1, kernel, 1)[..., kernel // 2]
        encoded = layer((values * visible.unsqueeze(-1)).transpose(1, 2)).transpose(1, 2)
        return encoded.masked_fill(~centers.unsqueeze(-1), 0), centers

    @staticmethod
    def similarity(features, visible, prototypes, ready):
        summary = F.normalize(masked_mean(features, visible), dim=-1)
        prototype_means = F.normalize(prototypes.mean(1), dim=-1)
        usable = visible.any(1, keepdim=True) & ready.unsqueeze(0)
        return (summary @ prototype_means.T).masked_fill(~usable, -1e4), usable

    def personalize(self, stream, context, classes, prototypes):
        projection = getattr(self.core, f'{stream}_srcproj')(context.detach())
        gamma = getattr(self.core, f'{stream}_gamma_mlp')(projection)
        beta = getattr(self.core, f'{stream}_beta_mlp')(projection)
        return gamma.unsqueeze(1) * prototypes[classes] + beta.unsqueeze(1)

    def repair_vision(self, text, text_mask, vision, vision_mask, classes=None):
        core = self.core
        if classes is None:
            text_similarity, _ = self.similarity(text, text_mask, core.proto_text, core.proto_ready_text)
            visual_similarity, _ = self.similarity(vision, vision_mask, core.proto_vision,
                                                   core.proto_ready_vision)
            classes = torch.maximum(text_similarity, visual_similarity).argmax(1)
        repaired = self.personalize('t2v', masked_mean(text, text_mask), classes, core.proto_vision)
        can_repair = core.proto_ready_vision[classes] & text_mask.any(1)
        fill = can_repair[:, None] & ~vision_mask
        return torch.where(fill.unsqueeze(-1), repaired, vision)

    def fuse(self, text, vision):
        core = self.core
        text, vision = text.transpose(0, 1), vision.transpose(0, 1)
        text_cross = core.trans_l_with_v(text, vision, vision)
        vision_cross = core.trans_v_with_l(vision, text, text)
        text_state = core.trans_l_mem(torch.cat((torch.zeros_like(text_cross), text_cross), dim=2))[-1]
        vision_state = core.trans_v_mem(torch.cat((vision_cross, torch.zeros_like(vision_cross)), dim=2))[-1]
        text_state = core.weight_l(text_state).sigmoid()
        vision_state = core.weight_v(vision_state).sigmoid()
        fused = torch.cat((text_state, vision_state, torch.zeros_like(text_state)), dim=1)
        update = core.proj2(F.dropout(F.relu(core.proj1(fused)), p=core.output_dropout,
                                      training=self.training))
        return core.out_layer(update + fused)

    def encode(self, values, masks):
        dropped = F.dropout(values[0], p=self.core.text_dropout, training=self.training)
        text, text_mask = self.project(dropped, masks[0], self.core.proj_l)
        vision, vision_mask = self.project(values[2], masks[2], self.core.proj_v)
        return text, text_mask, vision, vision_mask

    def forward(self, observed, visible):
        text, text_mask, vision, vision_mask = self.encode(observed, visible)
        repaired = self.repair_vision(text, text_mask, vision, vision_mask)
        forecast = self.fuse(text, repaired) / self.rating_scale
        return forecast, forecast.new_zeros(())

    def training_objective(self, observed, visible, full, native, intensity):
        core = self.core
        intensity = intensity * self.rating_scale
        classes = (intensity + 3).round().clamp(0, 6).long()
        text, text_mask, vision, vision_mask = self.encode(observed, visible)
        clean_vision, native_vision = self.project(full[2], native[2], core.proj_v)
        with torch.no_grad():
            labels = classes.cpu().tolist()
            for cache, features, mask in ((core._cache_text, text, text_mask),
                                          (core._cache_vision, clean_vision, native_vision)):
                available = mask.any(1).cpu().tolist()
                for label, present, value in zip(labels, available, features.detach().cpu(), strict=True):
                    if present:
                        cache[label].append(value)
        contrast = intensity.new_zeros(())
        for features, mask, prototypes, ready in (
            (text, text_mask, core.proto_text, core.proto_ready_text),
            (clean_vision, native_vision, core.proto_vision, core.proto_ready_vision),
        ):
            similarities, valid = self.similarity(features, mask, prototypes, ready)
            weight = valid.gather(1, classes.unsqueeze(1)).squeeze(1)
            loss = F.cross_entropy(similarities / core.infoNCET, classes, reduction='none')
            contrast = contrast + (loss * weight).sum() / weight.sum().clamp_min(1)
        reconstruction = intensity.new_zeros(())
        for stream, features, mask, context, context_mask, prototypes, ready in (
            ('t2v', clean_vision, native_vision, text, text_mask, core.proto_vision, core.proto_ready_vision),
            ('v2t', text, text_mask, clean_vision, native_vision, core.proto_text, core.proto_ready_text),
        ):
            personalized = self.personalize(stream, masked_mean(context, context_mask), classes, prototypes)
            error = (personalized - features.detach()).square().mean(-1)
            weight = mask & (ready[classes] & context_mask.any(1)).unsqueeze(1)
            reconstruction = reconstruction + (error * weight).sum() / weight.sum().clamp_min(1)
        repaired = self.repair_vision(text, text_mask, vision, vision_mask, classes)
        prediction = self.fuse(text, repaired).squeeze(-1)
        return (F.l1_loss(prediction, intensity) + .5 * self.contrastive_weight * contrast
                + .5 * self.reconstruction_weight * reconstruction)

    def finish_epoch(self):
        self.core.update_prototypes_epoch_end()


class AbsentAudio(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, features, mask):
        return features.new_zeros(features.shape[0], self.width)


class UCMRForVisualGaps(nn.Module):
    """Reuse the author's feature mode, residual gate, losses and distillation."""

    def __init__(self, upstream, dimensions, rating_scale, visual_only=True):
        super().__init__()
        package = upstream / 'UCMR-Net_code.zip'
        source = str(package / 'UCMR-Net_code' / 'src')
        sys.path.insert(0, source)
        try:
            from ucmr_net.models.ucmr_net import UCMRNet
            from ucmr_net.training.losses import add_distillation, compute_supervised_loss
        finally:
            sys.path.remove(source)

        with ZipFile(package) as archive:
            config = yaml.safe_load(archive.read('UCMR-Net_code/configs/mosei_ucmr.yaml'))
        model_config = config['model'] | {'text_mode': 'features'}
        self.core = UCMRNet(model_config, SimpleNamespace(text_mode='features', text_dim=dimensions[0],
                                                         audio_dim=dimensions[1], visual_dim=dimensions[2]))
        if visual_only:
            self.core.audio_encoder = AbsentAudio(model_config['hidden_dim'])
        self.rating_scale = rating_scale
        self.loss_config = config['loss']
        self.distillation_weight = config['training']['phases'][0]['distill_weight']
        self.supervised_loss = compute_supervised_loss
        self.add_distillation = add_distillation

    @staticmethod
    def inputs(observed, visible):
        text, audio, vision = observed
        text_mask, audio_mask, vision_mask = visible
        availability = torch.stack([mask.any(1) for mask in visible], dim=1).to(text.dtype)
        return dict(text_features=text, text_mask=text_mask, audio=audio, audio_mask=audio_mask,
                    visual=vision, visual_mask=vision_mask, availability_mask=availability)

    def forward(self, observed, visible):
        forecast = self.core(self.inputs(observed, visible))['regression'].unsqueeze(1) / self.rating_scale
        return forecast, forecast.new_zeros(())

    def training_objective(self, observed, visible, full, native, intensity):
        full_inputs = tuple(values * mask.unsqueeze(-1) for values, mask in zip(full, native, strict=True))
        batch = self.inputs(full_inputs, native)
        batch['label'] = intensity * self.rating_scale
        full_output = self.core(batch)
        losses = self.supervised_loss(full_output, batch, self.loss_config)
        masked_output = self.core(self.inputs(observed, visible))
        losses = self.add_distillation(losses, full_output, masked_output, self.distillation_weight)
        return losses['total']


def build_mosei_baseline(name, upstream, dimensions, lengths, corpus='mosei', visual_only=True):
    path = Path(upstream)
    # These models use seven ordinal bins on the MOSEI [-3, 3] rating scale.
    rating_scale = RATING_TO_MOSEI_SCALE[corpus]
    if name == 'api':
        return APIForVisualGaps(path, dimensions, lengths, rating_scale)
    return UCMRForVisualGaps(path, dimensions, rating_scale, visual_only=visual_only)
