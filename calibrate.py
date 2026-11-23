"""Fit scalar classification and intensity calibrations on the tuning split only."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax

from ablate import ranked_probability_score
from aggregate import f1_by_run


def scale_intensity(base, full, gain, scale):
    return np.tanh(base + gain * (full - base)) * scale


def fit_intensity_gain(base, full, target, scale):
    options = np.linspace(0., 1., 101)
    error = np.abs(scale_intensity(base, full, options[:, None, None], scale) - target)
    return float(options[error.mean(axis=(1, 2)).argmin()])


def fit_linear_score_gain(base, correction, target):
    options = np.linspace(0., 1., 101)
    error = np.abs(base + options[:, None, None] * correction - target)
    return float(options[error.mean(axis=(1, 2)).argmin()])


def temperature_scale(posterior, log_temperature):
    return softmax(np.log(posterior.clip(1e-12, 1)) / np.exp(log_temperature), axis=-1)


def log_loss(posterior, label):
    return float(-np.log(posterior[..., np.arange(len(label)), label].clip(1e-12, 1)).mean())


def fit_temperature(predicted, label):
    optimum = minimize_scalar(lambda parameter: log_loss(temperature_scale(predicted, parameter), label),
                              bounds=(np.log(.25), np.log(8)), method='bounded')
    if not optimum.success:
        raise ValueError('Tuning temperature failed.')
    return optimum.x


def fit_weight(predicted, anchor, label):
    def loss(weight):
        return log_loss(anchor + weight * (predicted - anchor), label)

    optimum = minimize_scalar(loss, bounds=(0, 1), method='bounded')
    if not optimum.success:
        raise ValueError('Tuning the mixture weight failed.')
    return min((0, optimum.x, 1), key=loss)


def scores(posterior, label):
    return dict(nll=log_loss(posterior, label),
                f1=float(f1_by_run(label, posterior[None]).mean()),
                rps=float(ranked_probability_score(posterior, label).mean()),
                brier=float(np.square(posterior - np.eye(3)[label]).sum(-1).mean()))


def calibrate(run):
    with np.load(run / 'tuning.npz', allow_pickle=False) as tune, \
            np.load(run / 'predictions.npz', allow_pickle=False) as evaluation:
        y_tune, y_eval = tune['y'], evaluation['y']
        base_tune, base_eval = tune['anchor'][1:3], evaluation['anchor'][1:3]
        base_temperature = fit_temperature(base_tune, y_tune)
        scaled_base_tune = temperature_scale(base_tune, base_temperature)
        scaled_base_eval = temperature_scale(base_eval, base_temperature)
        result = {}
        for arm in tune.files:
            if arm == 'y':
                continue
            raw_tune, raw_eval = tune[arm][1:3], evaluation[arm][1:3]
            temperature = fit_temperature(raw_tune, y_tune)
            scaled_tune = temperature_scale(raw_tune, temperature)
            weight = fit_weight(scaled_tune, scaled_base_tune, y_tune)
            scaled_eval = temperature_scale(raw_eval, temperature)
            mixture = scaled_base_eval + weight * (scaled_eval - scaled_base_eval)
            result[arm] = dict(temperature=float(np.exp(temperature)), fusion_weight=float(weight),
                               raw=scores(raw_eval, y_eval), scaled=scores(scaled_eval, y_eval),
                               mixture=scores(mixture, y_eval))
            print(f'{arm:14s} NLL raw={result[arm]["raw"]["nll"]:.4f} '
                  f'temp={result[arm]["scaled"]["nll"]:.4f} '
                  f'mix={result[arm]["mixture"]["nll"]:.4f}', flush=True)
    (run / 'calibration.json').write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    calibrate(parser.parse_args().run)
