import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import unittest
import numpy as np
from perturbations.salt_pepper import SaltPepperObservation, salt_and_pepper


class SaltPepperTests(unittest.TestCase):
    def test_probability_and_shared_rgb_channels(self):
        image = np.full((1000, 1000, 3), 127, dtype=np.uint8)
        output = salt_and_pepper(image, np.random.default_rng(42))
        white = np.all(output == 255, axis=-1).mean()
        black = np.all(output == 0, axis=-1).mean()
        self.assertAlmostEqual(white, 0.025, delta=0.002)
        self.assertAlmostEqual(black, 0.025, delta=0.002)
        np.testing.assert_array_equal(output[..., 0], output[..., 1])
        np.testing.assert_array_equal(image, 127)

    def test_both_views_fresh_and_reproducible_without_mutation(self):
        image = np.full((2, 1, 32, 32, 3), 127, dtype=np.uint8)
        info = {"pixels": image, "goal": image, "state": np.ones((2, 7))}
        shift = SaltPepperObservation(42)
        first, second = shift(info), shift(info)
        repeat = SaltPepperObservation(42)(info)
        for key in ("pixels", "goal"):
            np.testing.assert_array_equal(first[key], repeat[key])
            self.assertFalse(np.array_equal(first[key], second[key]))
            self.assertTrue(np.any(first[key] != 127))
        self.assertFalse(np.array_equal(first["pixels"], first["goal"]))
        np.testing.assert_array_equal(info["pixels"], 127)
        self.assertIs(first["state"], info["state"])

    def test_invalid_input_and_zero_noise(self):
        image = np.full((10, 10, 3), 127, dtype=np.uint8)
        np.testing.assert_array_equal(salt_and_pepper(image, np.random.default_rng(), 0), image)
        for value in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                salt_and_pepper(image, np.random.default_rng(), value)
        with self.assertRaises(ValueError):
            salt_and_pepper(image.astype(np.float32), np.random.default_rng())


class PushTProtocolTests(unittest.TestCase):
    def test_all_valid_rows_are_sampleable_and_goal_rows_match(self):
        from eval_pusht_suite import select_trials
        class Dataset:
            column_names = ["episode_idx", "step_idx", "state", "seed"]
            columns = {"episode_idx": np.array([0, 0, 0, 1, 1, 1]),
                       "step_idx": np.array([0, 1, 2, 0, 1, 2]),
                       "state": np.zeros((6, 7)), "seed": np.arange(6)}
            def get_col_data(self, key): return self.columns[key]
            def get_row_data(self, rows): return {k: v[rows] for k, v in self.columns.items()}
        trials = select_trials(Dataset(), 4, 1, 42)
        np.testing.assert_array_equal(trials["rows"], [0, 1, 3, 4])
        np.testing.assert_array_equal(trials["goal_rows"], [1, 2, 4, 5])

    def test_red_block_changes_rendering_but_not_dynamics(self):
        import stable_worldmodel as swm
        from eval_pusht_suite import reset_trial_world
        worlds = [swm.World("swm/PushT-v1", num_envs=1, image_shape=(224, 224), **kwargs)
                  for kwargs in ({}, {"init_value": {"block.color": np.array([255, 0, 0], dtype=np.uint8)}})]
        trials = {"states": np.array([[200, 200, 300, 300, 0.5, 0, 0]], dtype=np.float64),
                  "goal_states": np.array([[250, 250, 256, 256, 0.7, 0, 0]], dtype=np.float64),
                  "seeds": np.array([42])}
        try:
            for world in worlds:
                reset_trial_world(world, trials)
            for key in ("state", "goal_state"):
                np.testing.assert_allclose(worlds[0].infos[key], worlds[1].infos[key])
            for key in ("pixels", "goal"):
                default, red = worlds[0].infos[key], worlds[1].infos[key]
                self.assertFalse(np.array_equal(default, red))
                self.assertGreater(np.all(red == [255, 0, 0], axis=-1).sum(), 50)
                # Every changed render pixel belongs to the block/color edge.
                np.testing.assert_array_equal(np.all(default == 255, axis=-1), np.all(red == 255, axis=-1))
            for _ in range(3):
                results = [w.envs.step(np.array([[0.1, -0.1]], dtype=np.float32)) for w in worlds]
                np.testing.assert_allclose(results[0][-1]["state"], results[1][-1]["state"])
                np.testing.assert_array_equal(results[0][2], results[1][2])
        finally:
            for world in worlds: world.close()


if __name__ == "__main__":
    unittest.main()
