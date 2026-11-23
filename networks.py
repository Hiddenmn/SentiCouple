"""Text anchor and mask-aware audiovisual residual."""

from math import log

import torch
from torch import nn
from torch.nn import functional as F


class MaskedTemporalEncoder(nn.Module):
    def __init__(self, channels, width, mode='normalized'):
        super().__init__()
        self.mode = mode
        self.input = nn.Linear(channels, width)
        self.input_norm = nn.LayerNorm(width)
        if mode == 'pointwise':
            if width % 2:
                raise ValueError('A parameter-matched pointwise encoder requires an even width.')
            expanded = 3 * width // 2
            self.conv = nn.ModuleList(nn.Sequential(nn.Linear(width, expanded, bias=False), nn.GELU(),
                                                    nn.Linear(expanded, width)) for _ in range(2))
        else:
            self.conv = nn.ModuleList(nn.Conv1d(width, width, 3, padding=1) for _ in range(2))
        self.norm = nn.ModuleList(nn.LayerNorm(width) for _ in self.conv)
        if mode == 'normalized':
            self.register_buffer('density_kernel', torch.ones(1, 1, 3), persistent=False)

    def encode(self, values, visible):
        mask = visible.unsqueeze(-1)
        hidden = F.gelu(self.input_norm(self.input(values * mask))) * mask
        density = 1.
        if self.mode == 'normalized':
            count = F.conv1d(visible[:, None].float(), self.density_kernel, padding=1)
            density = count.transpose(1, 2).clamp_min(1) / 3
        for layer, norm in zip(self.conv, self.norm):
            if self.mode == 'pointwise':
                update = layer(hidden)
            else:
                update = F.conv1d(hidden.transpose(1, 2), layer.weight, padding=1).transpose(1, 2)
                update = update / density + layer.bias
            hidden = norm(hidden + F.gelu(update)) * mask
        return hidden

    def forward(self, values, visible, pooling='mean_max'):
        hidden = self.encode(values, visible)
        count = visible.sum(1)[:, None]
        mean = hidden.sum(1) / count.clamp_min(1)
        if pooling == 'mean':
            return torch.cat((mean, torch.zeros_like(mean)), -1)
        if pooling == 'mean_std':
            centered = (hidden - mean[:, None]) * visible[:, :, None]
            variance = centered.square().sum(1) / (count - 1).clamp_min(1)
            std = ((variance + 1e-6).sqrt() - 1e-3) * (count > 1)
            return torch.cat((mean, std), -1)
        maximum = hidden.masked_fill(~visible.unsqueeze(-1), -1e4).amax(1) * visible.any(1)[:, None]
        if pooling == 'max':
            mean = torch.zeros_like(mean)
        return torch.cat([mean, maximum], -1)


class TextQueryPooling(nn.Module):
    def __init__(self, bins, width, text_query=True):
        super().__init__()
        self.text_query = text_query
        self.position = nn.Parameter(torch.zeros(bins, width))
        self.attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)

    def forward(self, tokens, visible, context):
        if not self.text_query:
            context = tokens.sum(1) / visible.sum(1).clamp_min(1)[:, None]
        # The additional slot is accessible only for a fully absent stream.
        present = visible.any(-1, keepdim=True)
        placeholder = torch.zeros_like(tokens[:, :1])
        keys = torch.cat([tokens + self.position[None], placeholder], 1)
        values = torch.cat([tokens, placeholder], 1)
        visible = torch.cat([visible, ~present], 1)
        attended, _ = self.attention(context[:, None], keys, values,
                                     key_padding_mask=~visible, need_weights=False)
        average = tokens.sum(1) / visible[:, :-1].sum(1).clamp_min(1)[:, None] if self.text_query else context
        return torch.cat([average, attended.squeeze(1) * present], -1)


class TextAnchor(nn.Module):
    def __init__(self, input_dim, bins, width, visual_only=False):
        super().__init__()
        self.visual_only = visual_only
        self.text = MaskedTemporalEncoder(input_dim, width)
        observed_dims = bins if visual_only else 2 * bins
        self.context = nn.Sequential(nn.Linear(2 * width + observed_dims, width),
                                     nn.LayerNorm(width), nn.GELU())
        self.predictor = nn.Linear(width, 3)

    def forward(self, tokens, valid, audio_mask, vision_mask):
        content = self.text(tokens.float(), valid)
        observed = vision_mask if self.visual_only else torch.cat([audio_mask, vision_mask], -1)
        context = self.context(torch.cat([content, observed.float()], -1))
        return context, self.predictor(context)


class AudiovisualResidual(nn.Module):
    def __init__(self, audio_dim, vision_dim, width, attention_bins=None, independent_intensity=False,
                 text_query=True, visual_encoding='normalized', signed_only=False, logit_skip=True,
                 neutral_readout='softmax', visual_projection_only=False, visual_pooling='mean_max',
                 context_width=None):
        super().__init__()
        if context_width is None:
            context_width = width
        if attention_bins and visual_pooling != 'mean_max':
            raise ValueError('Alternative statistics require the fixed visual readout.')
        if attention_bins and context_width != width:
            raise ValueError('Attention requires equal context and fusion widths.')
        self.signed_only = signed_only
        self.logit_skip = logit_skip
        self.neutral_readout = neutral_readout
        self.visual_pooling = visual_pooling
        self.audio = MaskedTemporalEncoder(audio_dim, width) if audio_dim else None
        self.vision = MaskedTemporalEncoder(vision_dim, width, mode=visual_encoding)
        input_width = context_width + (4 if self.audio is not None else 2) * width
        self.predictor = nn.Sequential(nn.Linear(input_width, width), nn.LayerNorm(width),
                                       nn.GELU(), nn.Linear(width, 3))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        self.compatibility = nn.Linear(width, 1)
        self.audio_attention = TextQueryPooling(attention_bins, width, text_query) if attention_bins and audio_dim else None
        self.vision_attention = TextQueryPooling(attention_bins, width, text_query) if attention_bins else None
        self.intensity_head = nn.Linear(width, 1) if independent_intensity else None
        if self.intensity_head is not None:
            nn.init.zeros_(self.intensity_head.weight)
            nn.init.zeros_(self.intensity_head.bias)
        if visual_projection_only:
            # Initialize all common layers before removing the two visual blocks.
            self.vision.mode = 'projection'
            self.vision.conv = nn.ModuleList()
            self.vision.norm = nn.ModuleList()

    def forward(self, context, prior_logits, audio, vision, audio_mask, vision_mask,
                pairing=False, class_specific=True, return_intensity=False):
        vision_features = (self.vision(vision, vision_mask, self.visual_pooling)
                           if self.vision_attention is None else
                           self.vision_attention(self.vision.encode(vision, vision_mask), vision_mask, context))
        features, available = [context], vision_mask.any(1)
        if self.audio is not None:
            audio_features = self.audio(audio, audio_mask) if self.audio_attention is None else (
                self.audio_attention(self.audio.encode(audio, audio_mask), audio_mask, context))
            features.append(audio_features)
            available = available | audio_mask.any(1)
        features.append(vision_features)
        hidden = self.predictor[:-1](torch.cat(features, -1))
        change = self.predictor[-1](hidden)
        if self.signed_only:
            signed = (change[:, 2] - change[:, 0]) / 2
            change = torch.stack((-signed, torch.zeros_like(signed), signed), dim=-1)
        if self.logit_skip:
            masked_change = change * available[:, None]
            logits = prior_logits + masked_change
            if self.neutral_readout != 'softmax':
                logits = neutral_probability_update(logits, prior_logits, masked_change,
                                                    gradient_only=self.neutral_readout == 'gradient-only')
        else:
            logits = torch.where(available[:, None], change, prior_logits)
        if return_intensity:
            return logits, self.intensity_head(hidden).squeeze(-1) * available
        if pairing:
            shared = self.compatibility(hidden)
            if class_specific:
                return logits, shared + change - change.mean(-1, keepdim=True)
            return logits, shared.expand_as(change)
        return logits


def neutral_probability_update(logits, prior, change, gradient_only=False):
    """Make the neutral-to-polar log odds follow the affine neutral correction."""
    negative, neutral, positive = logits.unbind(dim=-1)
    updated_polar = torch.logaddexp(negative, positive)
    anchor_polar = torch.logaddexp(prior[:, 0], prior[:, 2])
    polar_center = (change[:, 0] + change[:, 2]) / 2
    correction = (updated_polar - anchor_polar) - polar_center
    if gradient_only:
        correction = correction - correction.detach()
    return torch.stack((negative, neutral + correction, positive), dim=-1)


def reference_loss(reference_probs, anchor_probs, variance_corrected):
    """Square of the mean displacement, with optional finite-sample correction."""
    displacement = reference_probs - anchor_probs[:, None]
    mean_squared = displacement.mean(1).square().sum(-1)
    if not variance_corrected:
        return mean_squared
    draws = displacement.shape[1]
    centered = displacement - displacement.mean(1, keepdim=True)
    correction = centered.square().sum(dim=(1, 2)) / (draws * (draws - 1))
    return mean_squared - correction


def sentiment_shift(logits, prior_logits):
    """The positive versus negative part of an audiovisual logit update."""
    residual = logits - prior_logits
    return (residual[:, 2] - residual[:, 0]) / 2


def mean_polarity_shift(logits, prior_logits):
    """Change in inverse-tanh mean polarity, with neutral shared between the two sides."""
    def log_odds(values):
        half_neutral = values[:, 1] - log(2)
        return torch.logaddexp(values[:, 2], half_neutral) - torch.logaddexp(values[:, 0], half_neutral)

    return (log_odds(logits) - log_odds(prior_logits)) / 2


def ordinal_readout(prior, shift):
    """Translate both cumulative logits by one shared sentiment update."""
    lower = prior[:, 0] - torch.logsumexp(prior[:, 1:], dim=-1)
    upper = torch.logsumexp(prior[:, :2], dim=-1) - prior[:, 2]
    gap = torch.log(-torch.expm1((lower - upper).clamp(max=-1e-7)))

    def log_probabilities(displacement):
        low, high = lower - displacement, upper - displacement
        return torch.stack([F.logsigmoid(low),
                            F.logsigmoid(high) + F.logsigmoid(-low) + gap,
                            F.logsigmoid(-high)], dim=-1)

    # The difference recovers the exact anchor logits at a zero shift.
    return prior + (log_probabilities(shift) - log_probabilities(0))


def intensity_anchor(prior, class_centers, correction=None):
    baseline = (prior.softmax(-1) @ class_centers).clamp(-.9999, .9999)
    logit = torch.atanh(baseline)
    return logit if correction is None else logit + correction


@torch.no_grad()
def fit_score_correction(context, prior, target, class_centers):
    """Fit a ridge correction using the frozen anchor outputs on fitting clips only."""
    features = context.flatten(0, 1).double()
    clipped_score = torch.atanh(target.clamp(-.95, .95))
    residual = (clipped_score[None, :] - intensity_anchor(prior, class_centers)).flatten().double()
    center = features.mean(0)
    scale = features.std(0, correction=0).clamp_min(1e-6)
    design = (features - center) / scale
    intercept = residual.mean()
    penalty = .01 * torch.eye(design.shape[1], device=design.device, dtype=design.dtype)
    gram = design.T @ design / len(design)
    rhs = design.T @ (residual - intercept) / len(design)
    weight = torch.linalg.solve(gram + penalty, rhs) / scale
    return weight.to(context.dtype), (intercept - center @ weight).to(context.dtype)


def intensity_readout(prior, shift, class_centers, correction=None):
    return torch.tanh(intensity_anchor(prior, class_centers, correction) + shift)


@torch.no_grad()
def intensity_gradient_weights(logits, label, intensity, target, visible, weight, beta):
    """Cap the opposing auxiliary derivative at half the class derivative."""
    probability = logits.softmax(dim=-1)
    class_gradient = probability[:, 2] - probability[:, 0] - (label - 1)
    score_gradient = weight * ((intensity - target) / beta).clamp(-1., 1.) * (1. - intensity.square())
    magnitude = score_gradient.abs()
    conflicts = class_gradient * score_gradient < 0
    cap = .5 * class_gradient.abs()
    fractions = (cap / magnitude.clamp_min(torch.finfo(magnitude.dtype).tiny)).clamp(max=1.)
    clip_weights = torch.where(conflicts, fractions, 1.)
    budget = magnitude * visible
    total = budget.sum()
    uniform = torch.where(total > 0, (budget * clip_weights).sum() / total, 1.)
    return clip_weights, uniform


def latent_intensity_loss(shift, prior, class_centers, target, label, visible, centered, correction=None):
    scale_error = torch.atanh(target.clamp(-.95, .95))
    error = intensity_anchor(prior, class_centers, correction) + shift - scale_error
    if centered:
        membership = F.one_hot(label, num_classes=3).to(error.dtype) * visible[:, None]
        class_means = (membership.T @ error) / membership.sum(0).clamp_min(1)
        error = error - class_means[label]
    losses = F.smooth_l1_loss(error, torch.zeros_like(error), beta=.15, reduction='none')
    return (losses * visible).sum() / visible.sum().clamp_min(1)
