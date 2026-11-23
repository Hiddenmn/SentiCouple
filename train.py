"""Fit the anchor and matched fusion comparisons, then save held-out predictions."""

import argparse
import copy
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.nn import functional as F

from calibrate import fit_intensity_gain, fit_linear_score_gain, scale_intensity
from conditional_intensity import ConditionalIntensity, MarginalIntensity
from data import load_clips, normalize_observed, observation_views, reference_banks
from networks import (AudiovisualResidual, TextAnchor, fit_score_correction, intensity_anchor,
                      intensity_gradient_weights, intensity_readout, latent_intensity_loss,
                      mean_polarity_shift, ordinal_readout, reference_loss, sentiment_shift)


ARMS = ['fusion', 'shrinkage', 'mean', 'moment', 'random_moment']
PAIRING_ARMS = ['pairing', 'random_pairing', 'shared_pairing']
ORDINAL_ARMS = ['ordered', 'intensity', 'ordered_intensity', 'independent_intensity']
CLASS_RPS_CONTROLS = ('fusion_rps', 'fusion_ce_rps')
NEUTRAL_READOUT_ARMS = {
    'intensity_factorized': 'factorized',
    'independent_factorized': 'factorized',
    'intensity_class_mean_factorized': 'factorized',
    'independent_class_mean_factorized': 'factorized',
    'intensity_gradient_decoupled': 'gradient-only',
    'intensity_class_mean_gradient_decoupled': 'gradient-only',
}
VISUAL_COMPONENT_ARMS = {
    'intensity_mean_pool': (False, 'mean'),
    'intensity_max_pool': (False, 'max'),
    'intensity_projection': (True, 'mean_max'),
    'intensity_projection_mean': (True, 'mean'),
    'intensity_projection_max': (True, 'max'),
    'independent_projection': (True, 'mean_max'),
    'fusion_projection': (True, 'mean_max'),
    'intensity_class_mean_projection': (True, 'mean_max'),
    'independent_class_mean_projection': (True, 'mean_max'),
}
CLASS_MEAN_ARMS = ('intensity_class_mean', 'independent_class_mean',
                   'intensity_class_mean_factorized', 'independent_class_mean_factorized',
                   'intensity_class_mean_gradient_decoupled',
                   'intensity_class_mean_projection', 'independent_class_mean_projection')
MARGINAL_INTENSITY_ARMS = {
    'tilt_exact_shared': 'exact',
    'tilt_exact_independent': 'exact',
    'tilt_tangent_shared': 'tangent',
    'tilt_tangent_independent': 'tangent',
}
CONDITIONAL_INTENSITY_ARMS = ('conditional_shared', 'conditional_independent', 'conditional_shuffled',
                             *MARGINAL_INTENSITY_ARMS)
DISTILLATION_ARMS = {
    'distill_native': (True, 'absolute'),
    'distill_matched': (False, 'absolute'),
    'distill_native_prior': (True, 'prior-corrected'),
    'distill_anchored_native': (True, 'anchored'),
    'distill_anchored_matched': (False, 'anchored'),
    'distill_anchored_independent': (True, 'anchored'),
}
INDEPENDENT_SCORE_ARMS = ('independent_intensity', 'independent_vision', 'independent_class_mean',
                          'independent_factorized', 'independent_class_mean_factorized',
                          'independent_projection', 'independent_class_mean_projection', 'conditional_independent',
                          'distill_anchored_independent', 'tilt_exact_independent', 'tilt_tangent_independent')
GRADIENT_CONTROLS = ('intensity_capped', 'intensity_uniform')
INTENSITY_CONTROLS = ['intensity_shuffled', 'intensity_latent', 'intensity_centered',
                      'intensity_mean_polarity', 'intensity_class_mean', 'independent_class_mean',
                      *NEUTRAL_READOUT_ARMS, *GRADIENT_CONTROLS]
ENCODER_ABLATIONS = {'intensity_pointwise': 'pointwise', 'intensity_unscaled': 'unscaled'}
ARCHITECTURE_CONTROLS = [*ENCODER_ABLATIONS, 'intensity_stream_query', 'intensity_signed_only',
                         'intensity_direct_logits']
STREAM_ABLATIONS = {
    'intensity_audio': 'vision', 'intensity_vision': 'audio',
    'fusion_vision': 'audio', 'independent_vision': 'audio',
}


class BatchSource:
    def __init__(self, clips, views, fit, device, visual_only=False, stream_text=False):
        self.views = torch.from_numpy(views).to(device)
        text_device = 'cpu' if stream_text else device
        self.text = torch.from_numpy(clips.text).to(text_device)
        self.text_mask = torch.from_numpy(clips.text_mask).to(text_device)
        self.audio = (torch.empty((*clips.audio.shape[:2], 0), device=device) if visual_only else
                      torch.from_numpy(normalize_observed(clips.audio, clips.audio_mask, fit)).to(device))
        self.vision = torch.from_numpy(normalize_observed(clips.vision, clips.vision_mask, fit)).to(device)
        self.label = torch.from_numpy(clips.label).to(device)
        self.intensity_scale = float(max(np.max(np.abs(clips.sentiment[fit])), 1e-6))
        self.intensity = torch.from_numpy(clips.sentiment / self.intensity_scale).to(device)
        centers = [clips.sentiment[fit][clips.label[fit] == y] for y in range(3)]
        self.class_centers = torch.tensor([x.mean() / self.intensity_scale if len(x) else 0 for x in centers],
                                          dtype=torch.float32, device=device)
        self.device = device

    def masks(self, view, clips):
        audio, vision = self.views[view, :, clips].unbind(1)
        return audio, vision

    def anchor(self, model, view, clips):
        text_index = clips.to(self.text.device)
        tokens = self.text[text_index].to(self.device)
        visible = self.text_mask[text_index].to(self.device)
        return model(tokens, visible, *self.masks(view, clips))

    def fusion(self, model, context, prior, view, clips, values_from, omit_stream=None, **kwargs):
        audio_mask, vision_mask = self.masks(view, clips)
        if omit_stream == 'audio':
            audio_mask = torch.zeros_like(audio_mask)
        elif omit_stream == 'vision':
            vision_mask = torch.zeros_like(vision_mask)
        return model(context, prior, self.audio[values_from], self.vision[values_from],
                     audio_mask, vision_mask, **kwargs)

    def references(self, model, context, prior, view, clips, values_from, ordered=False, **kwargs):
        repeats = values_from.shape[1]
        repeated_prior = prior.repeat_interleave(repeats, 0)
        logits = self.fusion(model, context.repeat_interleave(repeats, 0), repeated_prior,
                             view.repeat_interleave(repeats), clips.repeat_interleave(repeats),
                             values_from.reshape(-1), **kwargs)
        if kwargs.get('pairing'):
            return logits[1].reshape(len(clips), repeats, 3)
        if ordered:
            logits = ordinal_readout(repeated_prior, sentiment_shift(logits, repeated_prior))
        return logits.reshape(len(clips), repeats, 3).softmax(-1)


def index_batches(indices, batch_size, device, fixed_view):
    for start in range(0, len(indices), batch_size):
        batch = torch.as_tensor(indices[start:start + batch_size], device=device)
        yield torch.full_like(batch, fixed_view), batch


def macro_f1(label, prediction):
    return float(f1_score(label, prediction, labels=[0, 1, 2], average='macro', zero_division=0))


def selection_score(label, probabilities, criterion):
    if criterion == 'rps':
        cumulative = probabilities.cumsum(-1)[..., :2]
        targets = (label[:, None] <= np.arange(2)).astype(np.float32)
        return -float(np.square(cumulative - targets).mean())
    return macro_f1(label, probabilities.argmax(-1))


def ranked_probability_loss(logits, label):
    cumulative = logits.softmax(-1).cumsum(-1)[..., :-1]
    thresholds = torch.arange(cumulative.shape[-1], device=logits.device)
    targets = (label[:, None] <= thresholds).to(cumulative.dtype)
    return (cumulative - targets).square().mean()


def fitting_view(source, clips, rng, sampler):
    if sampler is not None:
        return sampler(rng, clips)
    return torch.as_tensor(rng.integers(len(source.views), size=len(clips)), device=source.device)


def fit_anchor(args, source, split, label, tuning_views=(1, 2), *, training_view_sampler=None):
    torch.manual_seed(args.seed)
    anchor = TextAnchor(source.text.shape[-1], args.bins, args.width,
                        visual_only=args.inputs == 'visual').to(source.device)
    optimizer = torch.optim.AdamW(anchor.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best_score, best_epoch, best = -float('inf'), 0, None
    for epoch in range(1, args.anchor_epochs + 1):
        anchor.train()
        fit = rng.permutation(split['fit'])
        for _, clips in index_batches(fit, args.batch, source.device, 0):
            view = fitting_view(source, clips, rng, training_view_sampler)
            loss = F.cross_entropy(source.anchor(anchor, view, clips)[1], source.label[clips])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        anchor.eval()
        with torch.inference_mode():
            scores = []
            for severity in tuning_views:
                predictions = [source.anchor(anchor, view, clips)[1].softmax(-1).cpu().numpy()
                               for view, clips in index_batches(split['tune'], args.batch, source.device, severity)]
                scores.append(selection_score(label[split['tune']], np.concatenate(predictions), args.selection))
        score = float(np.mean(scores))
        if score > best_score:
            best_score, best_epoch, best = score, epoch, copy.deepcopy(anchor.state_dict())
    anchor.load_state_dict(best)
    print(f'anchor epoch={best_epoch} tuning {args.selection} score={best_score:.4f}', flush=True)
    return anchor, dict(tune_score=best_score, criterion=args.selection, epoch=best_epoch)


def anchor_outputs(model, source, batch_size):
    model.eval()
    outputs = []
    with torch.no_grad():
        indices = np.arange(len(source.text))
        for severity in range(len(source.views)):
            batches = [source.anchor(model, view, clips)
                       for view, clips in index_batches(indices, batch_size, source.device, severity)]
            outputs.append(tuple(torch.cat(values) for values in zip(*batches)))
    return tuple(torch.stack(values) for values in zip(*outputs))


def shuffle_intensity(source, fit, label, rng):
    shuffled = source.intensity.clone()
    for category in range(3):
        indices = fit[label[fit] == category]
        shuffled[indices] = source.intensity[rng.permutation(indices)]
    return shuffled


def visual_component_options(arm):
    projection_only, pooling = VISUAL_COMPONENT_ARMS.get(arm, (False, 'mean_max'))
    return dict(visual_projection_only=projection_only, visual_pooling=pooling)


def read_teacher(path, seed, ids, masks, prior):
    with np.load(path, allow_pickle=False) as packet:
        if packet['seed'].item() != seed:
            raise ValueError('The teacher and student training seeds must match.')
        np.testing.assert_array_equal(packet['ids'], ids, err_msg='Teacher fitting clips')
        np.testing.assert_array_equal(packet['masks'], masks, err_msg='Teacher fitting views')
        logits = packet['logits']
        if logits.shape != (len(masks), len(ids), prior.shape[-1]) or not np.isfinite(logits).all():
            raise ValueError('The teacher needs finite logits for every fitting clip and view.')
        metadata = {key: packet[key].item() for key in ('teacher_run', 'revision')}
    return torch.as_tensor(logits, device=prior.device, dtype=prior.dtype), metadata


def distillation_targets(arm, teacher, prior, labels, fit):
    native, transform = DISTILLATION_ARMS[arm]
    logits = teacher[:1].expand_as(teacher) if native else teacher
    fit_labels = labels[fit]
    if transform == 'anchored':
        logits = prior[:, fit] + (logits - teacher[-1])
    elif transform == 'prior-corrected':
        counts = torch.bincount(fit_labels, minlength=prior.shape[-1]).clamp_min(1)
        logits = logits + counts.to(logits.dtype).log()
    target = torch.zeros_like(prior)
    target[:, fit] = .5 * (F.one_hot(fit_labels, prior.shape[-1]) + logits.softmax(-1))
    return target


def fit_fusion(args, arm, source, split, saved_anchor, label, banks, score_corrections=None,
               intensity_reference=None, teacher_targets=None, score_weights=None, class_objective=None,
               tuning_views=(1, 2), *, fusion_width=None, visual_pooling=None, training_view_sampler=None):
    torch.manual_seed(args.seed + 1)
    independent_intensity = arm in INDEPENDENT_SCORE_ARMS
    width = args.width if fusion_width is None else fusion_width
    pooling_options = visual_component_options(arm)
    if visual_pooling is not None:
        pooling_options['visual_pooling'] = visual_pooling
    model = AudiovisualResidual(source.audio.shape[-1], source.vision.shape[-1], width,
                                attention_bins=args.bins if args.backbone != 'temporal' else None,
                                independent_intensity=independent_intensity,
                                text_query=args.backbone == 'attention' and arm != 'intensity_stream_query',
                                visual_encoding=ENCODER_ABLATIONS.get(arm, args.visual_encoding),
                                signed_only=arm == 'intensity_signed_only',
                                logit_skip=arm != 'intensity_direct_logits',
                                neutral_readout=NEUTRAL_READOUT_ARMS.get(arm, 'softmax'),
                                context_width=args.width, **pooling_options).to(source.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng, donor_rng = np.random.default_rng(args.seed + 1), np.random.default_rng(args.seed + 2)
    contexts, priors = saved_anchor
    best_score, best_epoch, best = -float('inf'), 0, None
    banks = torch.as_tensor(banks, device=source.device)
    pairing = arm in PAIRING_ARMS
    class_specific = arm != 'shared_pairing'
    ordered = arm in ('ordered', 'ordered_intensity')
    score_shift = mean_polarity_shift if arm == 'intensity_mean_polarity' else sentiment_shift
    score_supervision = independent_intensity or arm in (
        'intensity', 'ordered_intensity', 'intensity_audio', 'intensity_vision',
        'intensity_shuffled', 'intensity_mean_polarity', 'intensity_class_mean',
        *ARCHITECTURE_CONTROLS, *GRADIENT_CONTROLS, *NEUTRAL_READOUT_ARMS, *DISTILLATION_ARMS) or (
            arm in VISUAL_COMPONENT_ARMS and arm.startswith('intensity'))
    latent_supervision = arm in ('intensity_latent', 'intensity_centered')
    gradient_control = arm in GRADIENT_CONTROLS
    gradient_stats = {}
    score_rng = np.random.default_rng(args.seed + 3)
    omit_stream = STREAM_ABLATIONS.get(arm)
    score_target = source.class_centers[source.label] if arm in CLASS_MEAN_ARMS else source.intensity
    for epoch in range(1, args.fusion_epochs + 1):
        model.train()
        gradient_counts = []
        if arm in ('intensity_shuffled', 'conditional_shuffled'):
            score_target = shuffle_intensity(source, split['fit'], label, score_rng)
        for _, clips in index_batches(rng.permutation(split['fit']), args.batch, source.device, 0):
            view = fitting_view(source, clips, rng, training_view_sampler)
            context, prior = contexts[view, clips], priors[view, clips]
            predicted = source.fusion(model, context, prior, view, clips, clips,
                                      pairing=pairing, class_specific=class_specific,
                                      return_intensity=independent_intensity, omit_stream=omit_stream)
            if pairing:
                predicted, real_score = predicted
            if independent_intensity:
                predicted, shift = predicted
            else:
                shift = score_shift(predicted, prior)
            if ordered:
                predicted = ordinal_readout(prior, shift)
            if intensity_reference is not None:
                predicted, shift = intensity_reference.readout(predicted, prior, shift)
            class_target = source.label[clips] if teacher_targets is None else teacher_targets[view, clips]
            loss = (F.cross_entropy(predicted, class_target) if class_objective is None else
                    class_objective(predicted, class_target, view, clips))
            if arm in CLASS_RPS_CONTROLS:
                ordinal_loss = ranked_probability_loss(predicted, source.label[clips])
                loss = ordinal_loss if arm == 'fusion_rps' else loss + args.intensity_weight * ordinal_loss
            correction = None if score_corrections is None else score_corrections[view, clips]
            if intensity_reference is not None:
                loss = loss + args.intensity_weight * intensity_reference.loss(
                    shift, source.label[clips], score_target[clips])
            elif score_supervision:
                intensity = intensity_readout(prior, shift, source.class_centers, correction)
                if gradient_control:
                    audio_mask, vision_mask = source.masks(view, clips)
                    visible = audio_mask.any(-1) | vision_mask.any(-1)
                    clip_weight, uniform_weight = intensity_gradient_weights(
                        predicted, source.label[clips], intensity, score_target[clips], visible,
                        args.intensity_weight, .15)
                    chosen = clip_weight if arm == 'intensity_capped' else uniform_weight
                    score_loss = F.smooth_l1_loss(intensity, score_target[clips], beta=.15, reduction='none')
                    loss = loss + args.intensity_weight * (chosen * score_loss).mean()
                    gradient_counts.append(torch.stack([(visible & (clip_weight < 1)).sum(), visible.sum()]))
                elif score_weights is not None:
                    individual = F.smooth_l1_loss(intensity, score_target[clips], beta=.15, reduction='none')
                    loss = loss + args.intensity_weight * (score_weights[clips] * individual).mean()
                else:
                    loss = loss + args.intensity_weight * F.smooth_l1_loss(intensity, score_target[clips], beta=.15)
            if latent_supervision:
                audio_mask, vision_mask = source.masks(view, clips)
                visible = audio_mask.any(-1) | vision_mask.any(-1)
                loss = loss + args.intensity_weight * latent_intensity_loss(
                    shift, prior, source.class_centers, score_target[clips], source.label[clips],
                    visible, centered=arm == 'intensity_centered', correction=correction)
            if arm == 'shrinkage':
                drift = (predicted.softmax(-1) - prior.softmax(-1)).square().sum(-1)
                loss = loss + args.strength * drift.mean()
            elif pairing or arm in ('mean', 'moment', 'random_moment'):
                bank = banks[view, clips]
                sizes = (bank >= 0).sum(-1)
                eligible = sizes > 0
                uniforms = torch.as_tensor(donor_rng.random((len(clips), args.draws)),
                                           dtype=torch.float64, device=source.device)
                choices = (uniforms[eligible] * sizes[eligible, None]).long()
                donors = bank[eligible].gather(1, choices).long()
                if len(donors):
                    reference = source.references(model, context[eligible], prior[eligible],
                                                  view[eligible], clips[eligible], donors,
                                                  pairing=pairing, class_specific=class_specific)
                    if pairing:
                        target = source.label[clips[eligible]]
                        positive = real_score[eligible].gather(1, target[:, None]).squeeze(1)
                        negative = reference.gather(
                            2, target[:, None, None].expand(-1, args.draws, 1)).squeeze(2)
                        penalty = (F.softplus(-positive) + F.softplus(negative).mean(1)) / 2
                        loss = loss + args.pairing_weight * penalty.sum() / len(clips)
                    else:
                        penalty = reference_loss(reference, prior[eligible].softmax(-1), arm != 'mean')
                        loss = loss + args.strength * penalty.sum() / len(clips)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scored, _, _, _ = predict_fusion(model, source, split['tune'], saved_anchor, args.batch,
                                        conditions=tuning_views, ordered=ordered, omit_stream=omit_stream,
                                        score_shift=score_shift, score_corrections=score_corrections,
                                        intensity_reference=intensity_reference)
        score = float(np.mean([selection_score(label[split['tune']], probs, args.selection) for probs in scored]))
        if score > best_score:
            best_score, best_epoch, best = score, epoch, copy.deepcopy(model.state_dict())
            if gradient_control:
                clipped, visible = torch.stack(gradient_counts).sum(0).tolist()
                gradient_stats = dict(clipped_fraction=clipped / max(visible, 1))
    model.load_state_dict(best)
    print(f'{arm} epoch={best_epoch} tuning {args.selection} score={best_score:.4f}', flush=True)
    return model, dict(tune_score=best_score, criterion=args.selection, epoch=best_epoch,
                       parameters=sum(p.numel() for p in model.parameters()), **gradient_stats)


@torch.inference_mode()
def predict_fusion(model, source, indices, saved_anchor, batch_size, conditions, banks=None,
                   ordered=False, omit_stream=None, score_shift=sentiment_shift, score_corrections=None,
                   return_score_logits=False, intensity_reference=None):
    model.eval()
    contexts, priors = saved_anchor
    probabilities, references, reference_log, intensity = [], [], [], []
    independent_intensity = model.intensity_head is not None
    for severity in conditions:
        round_real, round_ref, round_ref_log, round_intensity = [], [], [], []
        for view, clips in index_batches(indices, batch_size, source.device, severity):
            context, prior = contexts[view, clips], priors[view, clips]
            logits = source.fusion(model, context, prior, view, clips, clips,
                                   return_intensity=independent_intensity, omit_stream=omit_stream)
            if independent_intensity:
                logits, shift = logits
            else:
                shift = score_shift(logits, prior)
            if intensity_reference is None:
                correction = None if score_corrections is None else score_corrections[view, clips]
                score_logit = intensity_anchor(prior, source.class_centers, correction) + shift
                score = score_logit if return_score_logits else score_logit.tanh() * source.intensity_scale
            else:
                logits, shift = intensity_reference.readout(logits, prior, shift)
                score = intensity_reference.score_components(logits, shift) * source.intensity_scale
            round_intensity.append(score.cpu().numpy())
            if ordered:
                logits = ordinal_readout(prior, shift)
            round_real.append(logits.softmax(-1).cpu().numpy())
            if banks is not None:
                donors = banks[view, clips].long()
                eligible = donors >= 0
                count = eligible.sum(-1, keepdim=True)
                probs = source.references(model, context, prior, view, clips, donors.clamp_min(0),
                                          ordered=ordered, omit_stream=omit_stream)
                mean = (probs * eligible[..., None]).sum(1) / count.clamp_min(1)
                mean = mean.masked_fill(count == 0, float('nan'))
                targets = source.label[clips, None, None].expand(-1, donors.shape[1], 1)
                logp = probs.gather(2, targets).squeeze(2).clamp_min(1e-8).log()
                mean_logp = (logp * eligible).sum(1) / count.squeeze(1).clamp_min(1)
                mean_logp = mean_logp.masked_fill(count.squeeze(1) == 0, float('nan'))
                round_ref.append(mean.cpu().numpy())
                round_ref_log.append(mean_logp.cpu().numpy())
        probabilities.append(np.concatenate(round_real))
        intensity.append(np.concatenate(round_intensity))
        if round_ref:
            references.append(np.concatenate(round_ref))
            reference_log.append(np.concatenate(round_ref_log))
    return np.stack(probabilities), np.asarray(references), np.asarray(reference_log), np.stack(intensity)


@torch.no_grad()
def calibrated_head_forecasts(model, source, clips, split, saved_anchor, batch_size):
    """Predict the matched gap views using the original class-mean score calibration."""
    tune, held = split['tune'], split['eval']
    anchor_score = intensity_anchor(saved_anchor[1], source.class_centers).cpu().numpy()
    _, _, _, tune_score = predict_fusion(model, source, tune, saved_anchor, batch_size, conditions=(1, 2),
                                         return_score_logits=True)
    posterior, _, _, score = predict_fusion(model, source, held, saved_anchor, batch_size, conditions=(1, 2, 3),
                                            return_score_logits=True)
    gain = fit_intensity_gain(anchor_score[1:3, tune], tune_score, clips.sentiment[tune], source.intensity_scale)
    score = scale_intensity(anchor_score[1:4, held], score, gain, source.intensity_scale)
    np.testing.assert_array_equal(posterior[-1], saved_anchor[1][3, held].softmax(-1).cpu().numpy())
    expected_score = np.tanh(anchor_score[3, held]) * source.intensity_scale
    np.testing.assert_allclose(score[-1], expected_score, atol=2e-6, rtol=3e-6)
    return posterior[:2], score[:2], gain


def verify_donors(clips, views, fit, bank):
    recipient_view, recipient, _ = np.nonzero(bank >= 0)
    donor = bank[bank >= 0]
    assert np.isin(donor, fit).all()
    assert (clips.groups[recipient] != clips.groups[donor]).all()
    assert (views[recipient_view, :, recipient] <= views[0, :, donor]).all()


def run(args):
    clock = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    clips, split = load_clips(args.data, args.bins, args.split, args.split_seed, args.fold,
                              group_reference=args.group_reference, exclude_eval=args.exclude_eval,
                              eval_groups=args.eval_groups, neutral_margin=args.neutral_margin)
    summaries = {role: dict(n=len(indexes), videos=len(np.unique(clips.groups[indexes])),
                            labels=np.bincount(clips.label[indexes], minlength=3).tolist())
                 for role, indexes in split.items()}
    print(f'{args.data.name}, {summaries}', flush=True)
    views = observation_views(clips, [.35, .65], args.mask_seed,
                              preserve_vision_gaps=getattr(args, 'preserve_vision_gaps', False))
    visual_only = args.inputs == 'visual'
    if visual_only:
        views[:, 0] = False
    if args.skip_reference_analysis:
        semantic = random = np.full((len(views), len(clips.ids), 1), -1, dtype=np.int32)
        coverage = {}
        donors = None
    else:
        semantic, random = reference_banks(clips, views, split['fit'], args.neighbors, args.mask_seed + 1)
        verify_donors(clips, views, split['fit'], semantic)
        verify_donors(clips, views, split['fit'], random)
        coverage = {role: (semantic[:, indexes, 0] >= 0).mean(axis=1).tolist()
                    for role, indexes in split.items()}
        print(f'matched reference coverage {coverage}', flush=True)
        donors = torch.from_numpy(semantic).to(device)
    source = BatchSource(clips, views, split['fit'], device, visual_only=visual_only,
                         stream_text=args.stream_text)
    heldout = split['eval']
    if args.output.exists():
        raise FileExistsError(f'Results already exist at {args.output}')
    args.output.mkdir(parents=True)
    anchor, statistics = fit_anchor(args, source, split, clips.label)
    saved_anchor = anchor_outputs(anchor, source, args.batch)
    posterior = saved_anchor[1][:, heldout].softmax(-1).cpu().numpy()
    result = dict(ids=clips.ids[heldout], groups=clips.groups[heldout], y=clips.label[heldout],
                  sentiment=clips.sentiment[heldout],
                  masks=views[:, :, heldout], anchor=posterior,
                  eligible=semantic[:, heldout, 0] >= 0)
    tuning = dict(y=clips.label[split['tune']],
                  anchor=saved_anchor[1][:, split['tune']].softmax(-1).cpu().numpy())
    selection = dict(anchor=statistics)
    torch.save(anchor.cpu().state_dict(), args.output / 'anchor.pt')
    score_corrections = None
    if args.score_anchor == 'continuous':
        contexts, priors = saved_anchor
        fit = torch.as_tensor(split['fit'], device=device)
        weight, bias = fit_score_correction(contexts[:, fit], priors[:, fit], source.intensity[fit],
                                            source.class_centers)
        score_corrections = contexts @ weight + bias
        base_logit = intensity_anchor(priors[:, heldout], source.class_centers)
        result['anchor_score'] = ((base_logit + score_corrections[:, heldout]).tanh()
                                  * source.intensity_scale).cpu().numpy()
        result['class_mean_anchor_score'] = (base_logit.tanh() * source.intensity_scale).cpu().numpy()
        torch.save(dict(weight=weight.cpu(), bias=bias.cpu(), class_centers=source.class_centers.cpu(),
                        intensity_scale=source.intensity_scale), args.output / 'score_anchor.pt')
    if args.calibrate_score_gain:
        score_base = intensity_anchor(saved_anchor[1], source.class_centers, score_corrections).cpu().numpy()
        tune_base = score_base[1:3, split['tune']]
        heldout_base = score_base[:, heldout]
        result['anchor_score'] = np.tanh(heldout_base) * source.intensity_scale
        selection['anchor']['class_centers'] = source.class_centers.cpu().tolist()
        selection['anchor']['intensity_scale'] = source.intensity_scale
    intensity_references = {}
    active_tilts = {MARGINAL_INTENSITY_ARMS.get(arm, 'conditional')
                    for arm in args.arms if arm in CONDITIONAL_INTENSITY_ARMS}
    fit_scores = (source.intensity[split['fit']], source.label[split['fit']], source.class_centers)
    for mode in active_tilts:
        intensity_references[mode] = (ConditionalIntensity(*fit_scores) if mode == 'conditional'
                                      else MarginalIntensity(*fit_scores, exact=mode == 'exact'))
    teacher, teacher_metadata = None, None
    if args.teacher is not None:
        fit = split['fit']
        teacher, teacher_metadata = read_teacher(args.teacher, args.seed, clips.ids[fit],
                                                 views[:, :, fit], saved_anchor[1])
    for arm in args.arms:
        bank = random if arm in ('random_moment', 'random_pairing') else semantic
        reference = (intensity_references[MARGINAL_INTENSITY_ARMS.get(arm, 'conditional')]
                     if arm in CONDITIONAL_INTENSITY_ARMS else None)
        targets = (distillation_targets(arm, teacher, saved_anchor[1], source.label, split['fit'])
                   if arm in DISTILLATION_ARMS else None)
        model, selection[arm] = fit_fusion(args, arm, source, split, saved_anchor, clips.label,
                                           bank, score_corrections=score_corrections,
                                           intensity_reference=reference, teacher_targets=targets)
        ordered = arm in ('ordered', 'ordered_intensity')
        omit_stream = STREAM_ABLATIONS.get(arm)
        score_shift = mean_polarity_shift if arm == 'intensity_mean_polarity' else sentiment_shift
        probabilities, references, reference_log, scores = predict_fusion(
            model, source, heldout, saved_anchor, args.batch, range(len(views)), donors,
            ordered=ordered, omit_stream=omit_stream, score_shift=score_shift,
            score_corrections=score_corrections, return_score_logits=args.calibrate_score_gain,
            intensity_reference=reference)
        result[arm] = probabilities
        if donors is not None:
            result[f'{arm}_reference'] = references
            result[f'{arm}_reference_log'] = reference_log
        tuning[arm], _, _, tune_score = predict_fusion(model, source, split['tune'], saved_anchor,
                                                      args.batch, range(len(views)), ordered=ordered,
                                                      omit_stream=omit_stream, score_shift=score_shift,
                                                      score_corrections=score_corrections,
                                                      return_score_logits=args.calibrate_score_gain,
                                                      intensity_reference=reference)
        if reference is not None:
            base, detail = scores[..., 0], scores[..., 1]
            result[f'{arm}_class_mean_score'] = base
            result[f'{arm}_uncalibrated_score'] = base + detail
            gain = 1.0
            if args.calibrate_score_gain:
                gain = fit_linear_score_gain(tune_score[1:3, :, 0], tune_score[1:3, :, 1],
                                             clips.sentiment[split['tune']])
                selection[arm]['score_gain'] = gain
            scores = base + gain * detail
        elif args.calibrate_score_gain:
            gain = fit_intensity_gain(tune_base, tune_score[1:3], clips.sentiment[split['tune']],
                                      source.intensity_scale)
            result[f'{arm}_uncalibrated_score'] = scale_intensity(heldout_base, scores, 1., source.intensity_scale)
            scores = scale_intensity(heldout_base, scores, gain, source.intensity_scale)
            selection[arm]['score_gain'] = gain
        result[f'{arm}_score'] = scores
        np.testing.assert_array_equal(result[arm][-1], posterior[-1])
        if reference is not None:
            expected = posterior[-1] @ source.class_centers.cpu().numpy() * source.intensity_scale
            np.testing.assert_allclose(scores[-1], expected, atol=2e-6, rtol=3e-6)
        elif 'anchor_score' in result:
            np.testing.assert_allclose(scores[-1], result['anchor_score'][-1], atol=2e-6, rtol=3e-6)
        torch.save(model.cpu().state_dict(), args.output / f'{arm}.pt')
    np.savez_compressed(args.output / 'predictions.npz', **result)
    np.savez_compressed(args.output / 'tuning.npz', **tuning)
    meta = dict(config={key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()},
                views=['native', 'block35', 'block65', 'no_av'], splits=summaries,
                selection=selection, donor_coverage=coverage,
                seconds=time.perf_counter() - clock,
                hardware=torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__)
    if teacher_metadata is not None:
        meta['teacher'] = teacher_metadata
    marginal = intensity_references.get('exact') or intensity_references.get('tangent')
    if marginal is not None:
        grid = torch.linspace(-2., 2., 81, device=device)
        meta['polar_marginal'] = dict(delta=grid.cpu().tolist(),
                                     exact=marginal.signed_marginal(grid).cpu().tolist(),
                                     scale=marginal.shift_scale)
    (args.output / 'run.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'finished in {meta["seconds"]:.1f}s', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--split', choices=['pilot', 'official', 'grouped', 'new-videos'], default='pilot')
    parser.add_argument('--fold', type=int, default=0, choices=range(5))
    parser.add_argument('--group-reference', type=Path,
                        help='Trusted feature packet defining previously seen source videos.')
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument('--exclude-eval', nargs='+',
                            help='Exclude source videos in these trusted packets from an official test split.')
    evaluation.add_argument('--eval-groups', type=Path,
                            help='Text file listing source videos to retain from an official test split.')
    parser.add_argument('--split-seed', default=1729, type=int)
    parser.add_argument('--mask-seed', default=1777, type=int)
    parser.add_argument('--seed', default=97, type=int)
    parser.add_argument('--bins', default=24, type=int)
    parser.add_argument('--neighbors', default=16, type=int)
    parser.add_argument('--draws', default=4, type=int)
    parser.add_argument('--width', default=48, type=int)
    parser.add_argument('--neutral-margin', default=0., type=float,
                        help='Absolute raw sentiment scores at or below this value are neutral.')
    parser.add_argument('--backbone', choices=['temporal', 'attention', 'stream-query'], default='temporal')
    parser.add_argument('--visual-encoding', choices=['normalized', 'unscaled', 'pointwise'], default='normalized')
    parser.add_argument('--inputs', choices=['audiovisual', 'visual'], default='audiovisual')
    parser.add_argument('--preserve-vision-gaps', action='store_true', default=argparse.SUPPRESS,
                        help='Keep the visual-only mask stream when adding audio.')
    parser.add_argument('--skip-reference-analysis', action='store_true')
    parser.add_argument('--stream-text', action='store_true', help='Transfer each text batch from RAM.')
    parser.add_argument('--batch', default=96, type=int)
    parser.add_argument('--anchor-epochs', default=18, type=int)
    parser.add_argument('--fusion-epochs', default=14, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--strength', default=2.0, type=float)
    parser.add_argument('--pairing-weight', default=.5, type=float)
    parser.add_argument('--intensity-weight', default=.5, type=float)
    parser.add_argument('--score-anchor', choices=['class-mean', 'continuous'], default='class-mean',
                        help='Train a continuous score correction on the frozen text and mask features.')
    parser.add_argument('--calibrate-score-gain', action='store_true',
                        help='Scale each fitted intensity shift by its tuning-set gain.')
    parser.add_argument('--teacher', type=Path, help='The paired teacher logits on the fitting clips.')
    parser.add_argument('--selection', choices=['macro_f1', 'rps'], default='macro_f1')
    parser.add_argument('--arms', default=ARMS, nargs='+',
                        choices=[*ARMS, *PAIRING_ARMS, *ORDINAL_ARMS, *CLASS_RPS_CONTROLS, *INTENSITY_CONTROLS,
                                 *ARCHITECTURE_CONTROLS, *STREAM_ABLATIONS, *VISUAL_COMPONENT_ARMS,
                                 *CONDITIONAL_INTENSITY_ARMS, *DISTILLATION_ARMS])
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    for name in ('bins', 'neighbors', 'width', 'batch', 'anchor_epochs', 'fusion_epochs', 'lr'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.draws < 2 or min(args.strength, args.pairing_weight, args.intensity_weight,
                            args.seed, args.mask_seed, args.split_seed) < 0:
        parser.error('Expected at least two draws and nonnegative seeds and loss weights')
    if not np.isfinite(args.neutral_margin) or args.neutral_margin < 0:
        parser.error('The neutral margin must be finite and nonnegative.')
    if args.backbone != 'temporal' and args.width % 4 != 0:
        parser.error('The attention width must be divisible by four.')
    if args.visual_encoding == 'pointwise' and args.width % 2 != 0:
        parser.error('The parameter-matched pointwise width must be even.')
    if args.split == 'new-videos' and args.group_reference is None:
        parser.error('The new-videos split requires --group-reference.')
    if args.inputs == 'visual' and STREAM_ABLATIONS.keys() & set(args.arms):
        parser.error('Stream ablations require audiovisual input.')
    if args.score_anchor == 'continuous' and set(args.arms).intersection(CONDITIONAL_INTENSITY_ARMS):
        parser.error('Conditional intensity uses the class-mean score anchor.')
    if bool(set(args.arms).intersection(DISTILLATION_ARMS)) != (args.teacher is not None):
        parser.error('Distillation comparisons require --teacher with matching logits.')
    if 'intensity_stream_query' in args.arms and args.backbone != 'attention':
        parser.error('The stream-query control replaces an attention model’s text query.')
    if (args.exclude_eval or args.eval_groups) and args.split != 'official':
        parser.error('--exclude-eval and --eval-groups apply to official partitions.')
    if args.skip_reference_analysis and set(args.arms).intersection(['mean', 'moment', 'random_moment', *PAIRING_ARMS]):
        parser.error('The requested training objective requires observation-matched references.')
    return args


if __name__ == '__main__':
    run(parse_args())
