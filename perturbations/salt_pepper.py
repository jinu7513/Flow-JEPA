"""Seeded pixel-wise salt-and-pepper corruption of RGB observations."""
import numpy as np


def salt_and_pepper(images, rng, amount=0.05, salt_fraction=0.5):
    """Select pixels with probability amount; replace all RGB channels together.

    Images must be uint8 with RGB in the last dimension. A selected white pixel
    on a white background may look unchanged; amount denotes selection probability.
    """
    images = np.asarray(images)
    if images.dtype != np.uint8 or images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError("Expected uint8 RGB images with shape (..., H, W, 3)")
    if not 0 <= amount <= 1 or not 0 <= salt_fraction <= 1:
        raise ValueError("amount and salt_fraction must be in [0, 1]")
    draw = rng.random(images.shape[:-1])
    output = images.copy()
    output[draw < amount * salt_fraction] = 255
    output[(draw >= amount * salt_fraction) & (draw < amount)] = 0
    return output


class SaltPepperObservation:
    """Fresh, independent masks for current and goal views at each policy call."""

    def __init__(self, seed, amount=0.05, salt_fraction=0.5):
        self.rng = np.random.default_rng(seed)
        self.amount = amount
        self.salt_fraction = salt_fraction

    def __call__(self, infos):
        result = infos.copy()
        for key in ("pixels", "goal"):
            result[key] = salt_and_pepper(
                infos[key], self.rng, self.amount, self.salt_fraction
            )
        return result
