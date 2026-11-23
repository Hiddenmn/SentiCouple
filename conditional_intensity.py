"""Class-conditional intensity distributions fitted from the training scores."""

import torch
from torch.nn.utils.rnn import pad_sequence

from networks import sentiment_shift


class ConditionalIntensity:
    """Exponentially tilt each class's empirical score distribution."""

    def __init__(self, intensity, label, centers):
        self.centers = centers
        deviations, log_mass = [], []
        for category, center in enumerate(centers):
            values, counts = intensity[label == category].unique(return_counts=True)
            if not values.numel():
                values, counts = center.reshape(1), torch.ones_like(center).reshape(1)
            deviations.append(values - center)
            log_mass.append((counts.to(intensity.dtype) / counts.sum()).log())
        self.deviations = pad_sequence(deviations, batch_first=True)
        self.log_mass = pad_sequence(log_mass, batch_first=True, padding_value=-torch.inf)
        self.mass = self.log_mass.softmax(dim=-1)

    def loss(self, shift, label, intensity):
        """Negative log likelihood relative to the fixed empirical distribution."""
        log_weight = self.log_mass[label] + shift[:, None] * self.deviations[label]
        observed_deviation = intensity - self.centers[label]
        return (log_weight.logsumexp(dim=-1) - shift * observed_deviation).mean()

    def score_components(self, logits, shift):
        """Return the marginal class-mean score and its within-class correction."""
        posterior = logits.softmax(dim=-1)
        log_weight = self.log_mass + shift[:, None, None] * self.deviations
        mass_change = log_weight.softmax(dim=-1) - self.mass
        class_correction = (mass_change * self.deviations).sum(dim=-1)
        return torch.stack((posterior @ self.centers, (posterior * class_correction).sum(dim=-1)), dim=-1)

    def readout(self, logits, prior, shift):
        return logits, shift


class MarginalIntensity(ConditionalIntensity):
    """Use the common score tilt's polar mass ratio as a signed logit correction."""

    def __init__(self, intensity, label, centers, *, exact):
        super().__init__(intensity, label, centers)
        negative, positive = intensity[label == 0], intensity[label == 2]
        if not negative.numel() or not positive.numel() or negative.max() >= positive.min():
            raise ValueError('The polar intensity supports must be nonempty and ordered.')
        self.shift_scale = 2.0 / float(centers[2] - centers[0])
        self.exact = exact
        self.polar_log_mass = self.log_mass[[0, 2]]
        self.polar_deviations = self.deviations[[0, 2]]
        self.polar_centers = self.centers[[0, 2]]
        self.zero_partition = self.polar_log_mass.logsumexp(dim=-1)

    def signed_marginal(self, signed):
        natural = signed * self.shift_scale
        log_weight = self.polar_log_mass + natural[:, None, None] * self.polar_deviations
        moment = (log_weight.logsumexp(dim=-1) - self.zero_partition
                  + natural[:, None] * self.polar_centers)
        return (moment[:, 1] - moment[:, 0]) / 2

    def readout(self, logits, prior, shift):
        if self.exact:
            signed = sentiment_shift(logits, prior)
            difference = self.signed_marginal(signed) - signed
            # The null update is exact and the marginal is tangent to the linear map there.
            difference = torch.where(signed == 0, 0., difference)
            logits = logits + torch.stack((-difference, torch.zeros_like(difference), difference), dim=-1)
        return logits, shift * self.shift_scale
