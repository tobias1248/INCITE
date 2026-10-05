from __future__ import annotations

import json

import numpy as np

from libct.global_real import GLOBAL_X_INPUT_NAME, build_aces_like_global_real_config
from libct.record import ConcolicTestRecorder


def test_aces_hybrid_stats_preserve_transform_parameters_and_solver_mode():
    seed = np.asarray([[[0.2, 0.4, 0.7]]], dtype=np.float64)
    recorder = ConcolicTestRecorder(None, "case_0")
    recorder.input_shape = seed.shape
    recorder.global_real_config = build_aces_like_global_real_config(seed, {
        "global_shift_kind": "aces-contrast", "effective_min": -0.1,
        "effective_max": 0.1, "bounds_mode": "clip", "pwl_max_segments": 8,
        "pwl_error_tolerance": 1.0 / 255.0, "hybrid_de_enabled": True,
    })
    recorder.extra_meta.update(
        hybrid_de_seed_x=0.06, hybrid_de_shift_kind="aces-brightness",
        hybrid_transform_order=["aces-brightness", "aces-contrast"],
        smt_path_mode="last", smt_duplicate_candidate_count=2,
    )
    inputs = {f"v_0_0_{i}": float(seed[0, 0, i]) for i in range(3)}
    inputs[GLOBAL_X_INPUT_NAME] = 0.02

    recorder.save_sat_input(inputs)
    recorder.find_adversarial_input(inputs, attack_label=1)
    stats = json.loads(json.dumps(recorder.output_stats_dict()))

    assert stats["meta"]["smt_path_mode"] == "last"
    assert stats["meta"]["smt_duplicate_candidate_count"] == 2
    assert stats["meta"]["hybrid_transform_order"] == ["aces-brightness", "aces-contrast"]
    for prefix in ("global_real_last_sat", "global_real_solved"):
        assert stats["meta"][prefix + "_de_x"] == 0.06
        assert stats["meta"][prefix + "_x"] == 0.02
        assert stats["meta"][prefix + "_pyct_shift_kind"] == "aces-contrast"
    assert recorder.global_real_sat_x == [0.02]
