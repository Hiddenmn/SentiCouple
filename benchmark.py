"""Train a published fusion architecture under the shared video-gap protocol."""

import argparse
import json
from pathlib import Path
import platform
import time

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
import torch
from torch.nn import functional as F

from ablate import ranked_probability_score
from data import load_clips, observation_views
from mosei_baselines import MOSEI_BASELINES
from networks import TextAnchor
from published_baselines import ANCHORED_HEADS, UPSTREAM, AnchoredReadout, build_published
from train import BatchSource, index_batches


def ordinal_probabilities(scores, calibration):
    center, scale, lower, log_gap, slope = calibration
    cutpoints = np.array([lower, lower + np.exp(log_gap)])
    cumulative = expit(cutpoints - slope * ((scores - center) / scale)[..., None])
    return np.stack((cumulative[..., 0], cumulative[..., 1] - cumulative[..., 0],
                     1 - cumulative[..., 1]), axis=-1)


def calibrate_intensity(scores, labels):
    """Fit an increasing proportional-odds readout by tuning-set RPS."""
    scores = scores.astype(np.float64)
    center, scale = scores.mean(), max(scores.std(), 1e-6)
    standardized = ((scores - center) / scale).reshape(-1, 1)
    observed = np.broadcast_to(labels <= np.arange(2)[:, None], (*scores.shape[:-1], 2, scores.shape[-1]))
    observed = np.swapaxes(observed, -1, -2).reshape(-1, 2)
    cutpoints = logit(observed.mean(axis=0).clip(1e-5, 1 - 1e-5))

    def objective(parameters):
        lower, log_gap, slope = parameters
        gap = np.exp(log_gap)
        cumulative = expit(np.array([lower, lower + gap]) - slope * standardized)
        errors = cumulative - observed
        gradient = 2 / errors.size * errors * cumulative * (1 - cumulative)
        return np.square(errors).mean(), np.array([
            gradient.sum(), gap * gradient[:, 1].sum(), -(standardized * gradient).sum()])

    initial = (cutpoints[0], np.log(max(cutpoints[1] - cutpoints[0], 1e-3)), .5)
    result = minimize(objective, initial, jac=True, method='L-BFGS-B',
                      bounds=[(-20, 20), (-10, 4), (0, 32)])
    if not result.success:
        raise RuntimeError(f'Ordinal calibration failed: {result.message}')
    return np.array([center, scale, *result.x])


def model_inputs(source, view, indices, reconstruct=False):
    text_index = indices.to(source.text.device)
    text = source.text[text_index].to(source.device).float()
    text_mask = source.text_mask[text_index].to(source.device)
    audio = source.audio[indices]
    if audio.shape[-1] == 0:
        audio = audio.new_zeros((*audio.shape[:-1], 1))
    vision = source.vision[indices]
    audio_mask, vision_mask = source.masks(view, indices)
    full = (text, audio, vision)
    visible = (text_mask, audio_mask, vision_mask)
    observed = tuple(values * mask[..., None] for values, mask in zip(full, visible))
    if reconstruct:
        return observed, visible, full, (text_mask, *source.masks(torch.zeros_like(view), indices))
    return observed, visible


def fit_batches(fit, batch):
    batches = [fit[start:start + batch] for start in range(0, len(fit), batch)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2:] = [np.concatenate(batches[-2:])]
    return batches


@torch.inference_mode()
def predict(model, source, indices, batch, conditions):
    model.eval()
    predictions = []
    for condition in conditions:
        rows = [model(*model_inputs(source, view, clips))[0].cpu().numpy()
                for view, clips in index_batches(indices, batch, source.device, condition)]
        predictions.append(np.concatenate(rows))
    result = np.stack(predictions)
    if not np.isfinite(result).all():
        raise ValueError('A published backbone produced nonfinite scores.')
    return result


def forecast(raw, head, scale, calibration):
    if head == 'regression':
        return ordinal_probabilities(raw[..., 0], calibration), raw[..., 0]
    probability = torch.from_numpy(raw[..., :3]).softmax(-1).numpy()
    return probability, np.tanh(raw[..., 3]) * scale


def run_config(args):
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
              if key != 'resume' and (key != 'neutral_margin' or value)}
    if config.get('rmnet_class_weighting') == 'inverse-frequency':
        config.pop('rmnet_class_weighting')
    return config


def save_training_state(path, args, model, optimizer, rng, best, history, seconds, device):
    temporary = path.with_suffix('.tmp')
    state = dict(config=run_config(args), model=model.state_dict(), optimizer=optimizer.state_dict(),
                 sampler=rng.bit_generator.state, torch_rng=torch.get_rng_state(), best=best,
                 history=history, training_seconds=seconds)
    if device.type == 'cuda':
        state['cuda_rng'] = torch.cuda.get_rng_state(device)
    torch.save(state, temporary)
    temporary.replace(path)


def fit_model(args, model, source, split, labels):
    optimizer_class = torch.optim.Adam if args.model == 'api' else torch.optim.AdamW
    optimizer = optimizer_class((p for p in model.parameters() if p.requires_grad), lr=args.lr,
                                weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 1)
    best = None
    tune_labels = labels[split['tune']]
    history = []
    seconds = 0.
    class_weights = None
    if args.model == 'rm-net':
        counts = torch.bincount(source.label[split['fit']], minlength=3).clamp_min(1)
        if args.rmnet_class_weighting == 'uniform':
            class_weights = torch.ones_like(counts, dtype=torch.float32)
        else:
            inverse = counts.float().reciprocal()
            class_weights = inverse / inverse.mean()
    checkpoint = args.output / 'resume.pt'
    if args.resume and checkpoint.exists():
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if state['config'] != run_config(args):
            raise ValueError('Resume requires the original training configuration.')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        rng.bit_generator.state = state['sampler']
        torch.set_rng_state(state['torch_rng'])
        if source.device.type == 'cuda':
            torch.cuda.set_rng_state(state['cuda_rng'], source.device)
        best, history, seconds = state['best'], state['history'], state['training_seconds']
        del state
        print(f'Resumed after epoch {len(history)}.', flush=True)
    for epoch in range(len(history) + 1, args.epochs + 1):
        if best is not None and epoch - 1 - best['epoch'] >= args.patience:
            break
        start = time.perf_counter()
        model.train()
        for indices in fit_batches(rng.permutation(split['fit']), args.batch):
            clips = torch.as_tensor(indices, device=source.device)
            view = torch.from_numpy(rng.integers(len(source.views), size=len(indices))).to(source.device)
            inputs = model_inputs(source, view, clips, reconstruct=True)
            if args.model in MOSEI_BASELINES:
                total_loss = model.training_objective(*inputs, source.intensity[clips] * source.intensity_scale)
            elif class_weights is not None:
                total_loss = model.training_loss(*inputs, source.label[clips], source.intensity[clips], class_weights)
            else:
                output, auxiliary = model(*inputs)
                if args.head == 'regression':
                    target = source.intensity[clips] * source.intensity_scale
                    loss = F.mse_loss if args.model == 'lnln' else F.l1_loss
                    task_loss = loss(output[:, 0], target)
                else:
                    task_loss = (F.cross_entropy(output[:, :3], source.label[clips]) +
                                 .5 * F.smooth_l1_loss(output[:, 3].tanh(), source.intensity[clips], beta=.15))
                total_loss = task_loss + auxiliary
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), .6 if args.model == 'api' else 1.)
            optimizer.step()
        if args.model == 'api':
            model.finish_epoch()
        tune = predict(model, source, split['tune'], args.batch, (1, 2))
        calibration = calibrate_intensity(tune[..., 0], tune_labels) if args.head == 'regression' else None
        posterior, _ = forecast(tune, args.head, source.intensity_scale, calibration)
        rps = float(ranked_probability_score(posterior, tune_labels).mean())
        history.append(rps)
        if best is None or rps < best['rps']:
            best = dict(epoch=epoch, rps=rps, calibration=calibration.tolist() if calibration is not None else None,
                        weights={name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
        elapsed = time.perf_counter() - start
        seconds += elapsed
        save_training_state(checkpoint, args, model, optimizer, rng, best, history, seconds, source.device)
        print(f'{args.model} {args.head} epoch={epoch} tuning RPS={rps:.6f} '
              f'best={best["rps"]:.6f} ({elapsed:.1f}s)', flush=True)
    model.load_state_dict(best.pop('weights'))
    return best, history, seconds


def verify_valence_reference(args, clips, split):
    reference = json.loads((args.match_run / 'run.json').read_text(encoding='utf-8'))
    config = reference['config']
    expected = dict(split='official', seed=args.seed, inputs=args.inputs, bins=24, fold=0,
                    split_seed=1729, mask_seed=1777, neutral_margin=.10, selection='rps',
                    group_reference=None, exclude_eval=None, eval_groups=None)
    if (any(config.get(key) != value for key, value in expected.items()) or
            config.get('preserve_vision_gaps', False) != getattr(args, 'preserve_vision_gaps', False) or
            Path(config['data']).resolve() != args.data.resolve()):
        raise ValueError('Use the same-seed compact forecast on the official valence split.')
    for role, indices in split.items():
        summary = dict(n=len(indices), videos=len(np.unique(clips.groups[indices])),
                       labels=np.bincount(clips.label[indices], minlength=3).tolist())
        if summary != reference['splits'][role]:
            raise ValueError(f'The {role} groups and label counts must match the compact reference.')
    with np.load(args.match_run / 'tuning.npz', allow_pickle=False) as tune:
        np.testing.assert_array_equal(clips.label[split['tune']], tune['y'])


def run(args, dataset=None):
    clock = time.perf_counter()
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.deterministic = True
    if dataset is None:
        dataset = load_clips(args.data, 24, args.split, 1729, group_reference=args.group_reference,
                             eval_groups=args.eval_groups, neutral_margin=args.neutral_margin)
    clips, split = dataset
    if args.corpus in ('omg', 'msp-face'):
        verify_valence_reference(args, clips, split)
    views = observation_views(clips, [.35, .65], 1777,
                              preserve_vision_gaps=getattr(args, 'preserve_vision_gaps', False))
    if args.inputs == 'visual':
        views[:, 0] = False
    heldout = split['eval']
    observed = dict(ids=clips.ids[heldout], groups=clips.groups[heldout], y=clips.label[heldout],
                    sentiment=clips.sentiment[heldout], masks=views[:, :, heldout])
    with np.load(args.match_run / 'predictions.npz', allow_pickle=False) as reference:
        for key, value in observed.items():
            np.testing.assert_array_equal(value, reference[key], err_msg=key)
    print(f'{len(split["fit"])} fit / {len(split["tune"])} tuning / {len(heldout)} test clips, '
          f'{len(np.unique(observed["groups"]))} test videos; inputs match {args.match_run.name}', flush=True)
    source = BatchSource(clips, views, split['fit'], torch.device(args.device),
                         visual_only=args.inputs == 'visual', stream_text=True)
    dimensions = (source.text.shape[-1], max(1, source.audio.shape[-1]), source.vision.shape[-1])
    lengths = (source.text.shape[1], source.audio.shape[1], source.vision.shape[1])
    torch.manual_seed(args.seed + 1)
    anchored = args.head in ANCHORED_HEADS
    head = 'joint' if anchored else args.head
    model = build_published(args.model, args.upstream, args.corpus, dimensions, lengths,
                            head, args.inputs == 'visual')
    if anchored:
        anchor_config = json.loads((args.match_run / 'run.json').read_text(encoding='utf-8'))['config']
        if anchor_config['inputs'] != args.inputs or anchor_config['seed'] != args.seed:
            raise ValueError('Pair the anchored readout with the same-input, same-seed text checkpoint.')
        with torch.random.fork_rng(devices=[]):
            anchor = TextAnchor(dimensions[0], anchor_config['bins'], anchor_config['width'],
                                visual_only=args.inputs == 'visual')
        anchor.load_state_dict(torch.load(args.match_run / 'anchor.pt', map_location='cpu', weights_only=True))
        model = AnchoredReadout(model, anchor, source.class_centers, shared=args.head == 'anchored-shared')
    model = model.to(source.device)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f'{args.model}: {parameters:,} trainable parameters', flush=True)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    best, history, training_seconds = fit_model(args, model, source, split, clips.label)
    raw = predict(model, source, heldout, args.batch, range(len(views)))
    posterior, score = forecast(raw, args.head, source.intensity_scale, best['calibration'])
    if anchored:
        with np.load(args.match_run / 'predictions.npz', allow_pickle=False) as reference:
            np.testing.assert_allclose(posterior[-1], reference['anchor'][-1], rtol=2e-5, atol=4e-6)
    np.savez_compressed(args.output / 'predictions.npz', **observed, posterior=posterior, score=score, raw=raw)
    tuning = predict(model, source, split['tune'], args.batch, range(len(views)))
    tune_probability, tune_score = forecast(tuning, args.head, source.intensity_scale, best['calibration'])
    np.savez_compressed(args.output / 'tuning.npz', y=clips.label[split['tune']],
                        posterior=tune_probability, score=tune_score, raw=tuning)
    torch.save(model.cpu().state_dict(), args.output / 'model.pt')
    result = dict(config=run_config(args),
                  upstream_revision=UPSTREAM[args.model], trainable_parameters=parameters, selection=best,
                  frozen_parameters=sum(parameter.numel() for parameter in model.parameters()
                                        if not parameter.requires_grad), training_seconds=training_seconds,
                  tune_rps_by_epoch=history, python=platform.python_version(), torch=str(torch.__version__),
                  seconds=time.perf_counter() - clock)
    result['heldout_rps'] = float(ranked_probability_score(posterior[1:3], observed['y']).mean())
    result['heldout_mae'] = float(np.abs(score[1:3] - observed['sentiment']).mean())
    metadata = args.output / 'run.json'
    temporary = metadata.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(metadata)
    (args.output / 'resume.pt').unlink(missing_ok=True)
    print(f'held-out RPS={result["heldout_rps"]:.6f}, MAE={result["heldout_mae"]:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=tuple(UPSTREAM), required=True)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--head', choices=['regression', 'joint', *ANCHORED_HEADS], default='regression')
    parser.add_argument('--corpus', choices=['sims', 'mosei', 'mosi', 'omg', 'msp-face'], required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--match-run', type=Path, required=True)
    parser.add_argument('--split', choices=['new-videos', 'official'], required=True)
    parser.add_argument('--group-reference', type=Path)
    parser.add_argument('--eval-groups', type=Path)
    parser.add_argument('--neutral-margin', type=float, default=0.)
    parser.add_argument('--inputs', choices=['visual', 'audiovisual'], default='visual')
    parser.add_argument('--preserve-vision-gaps', action='store_true', default=argparse.SUPPRESS,
                        help='Keep the visual-only mask stream when adding audio.')
    parser.add_argument('--seed', type=int, default=97)
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=.01)
    parser.add_argument('--rmnet-class-weighting', choices=('inverse-frequency', 'uniform'), default='inverse-frequency')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--resume', action='store_true', help='Continue the exact fit from its latest saved epoch.')
    args = parser.parse_args()
    if (args.output / 'run.json').exists() or (args.output.exists() and not args.resume):
        parser.error(f'Results already exist at {args.output}.')
    if args.split == 'new-videos' and args.group_reference is None:
        parser.error('Supply the previous feature packet via --group-reference.')
    if args.corpus in ('omg', 'msp-face') and (args.split != 'official' or args.neutral_margin != .10
                                             or args.eval_groups or args.group_reference):
        parser.error('Use the official valence split and pass --neutral-margin .10.')
    if args.model == 'rm-net' and args.head != 'joint':
        parser.error('RM-Net uses the published categorical and additive intensity heads (--head joint).')
    if args.model in MOSEI_BASELINES and (args.corpus, args.head, args.inputs) != ('mosei', 'regression', 'visual'):
        parser.error('Use --corpus mosei --head regression --inputs visual for this cached-feature adapter.')
    if args.model != 'rm-net' and args.rmnet_class_weighting != 'inverse-frequency':
        parser.error('--rmnet-class-weighting changes the classification terms of the RM-Net core.')
    if min(args.epochs, args.patience, args.threads) < 1 or args.batch < 2 or args.lr <= 0:
        parser.error('Epochs, patience, threads and learning rate must be positive. Batch size must be at least 2.')
    run(args)


if __name__ == '__main__':
    main()
