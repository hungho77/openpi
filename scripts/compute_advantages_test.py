import numpy as np

from . import compute_advantages


def test_compute_advantages():
    # One successful episode of length 4, task max length 4.
    table = {0: {"length": 4, "max_length": 4, "success": True}}
    episode_index = np.zeros(4, dtype=np.int64)
    frame_index = np.array([0, 1, 2, 3])
    # A perfect value function predicts the return, so all advantages are zero.
    values = np.array([-0.75, -0.5, -0.25, 0.0], dtype=np.float32)

    advantages, returns = compute_advantages.compute_advantages(
        values, episode_index, frame_index, table, n_step=2, failure_penalty=1.0
    )
    np.testing.assert_allclose(returns, values)
    np.testing.assert_allclose(advantages, 0.0, atol=1e-6)

    # Underestimating the value of frame 2 makes reaching it from frame 0 look bad, and frame 2 itself look good.
    values[2] = -0.5
    advantages, _ = compute_advantages.compute_advantages(
        values, episode_index, frame_index, table, n_step=2, failure_penalty=1.0
    )
    np.testing.assert_allclose(advantages, [-0.25, 0.0, 0.25, 0.0], atol=1e-6)
