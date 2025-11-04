"""
Compatibility layer exposing experiment runner utilities and task specifications.

New code should import directly from:
    - utils.experiment_runner
    - utils.experiment_task_specs
"""

from utils.experiment_runner import (  # noqa: F401
    GeneralRunner,
    QueueRunner,
    ShapRunner,
    run_multi_attack_subprocess_cpu_timeout,
    run_multi_attack_subprocess_wall_timeout,
    run_multi_attack_subprocess_wall_timeout_shap,
    run_multi_attack_subprocess_wall_timeout_task_queue,
    run_multi_attack_subprocess_general,
    run_multi_attack_subprocess_with_shap,
    run_multi_attack_subprocess_with_queue,
)
from utils.experiment_task_specs import (  # noqa: F401
    GenerationResult,
    QueueMode,
    TaskGenerationSpec,
    fashion_mnist_transformer_shap,
    fashion_mnist_transformer_shap_calculate_all,
    fashion_mnist_transformer_random,
    get_save_dir_from_save_exp,
    imdb_shap_1_2_3_4_8_range02,
    imdb_transformer_shap_1_2_3_4_8_range02,
    mnist_lstm_15_1_2_3_4_8_range02,
    mnist_lstm_1_2_3_4_8_range02,
    pyct_random_1_4_8_16_32,
    pyct_rnn_random_1_4_8_16_32,
    pyct_rnn_shap_1_4_8_16_32,
    pyct_shap_1_4_8_16_32,
    sentiment_lstm_lstm_15_1_2_3_4_8_range02,
    stock_random_1_2_3_4_8_range02,
    stock_shap_1_2_3_4_8_limit_range02,
)

__all__ = [
    "run_multi_attack_subprocess_wall_timeout",
    "run_multi_attack_subprocess_wall_timeout_shap",
    "run_multi_attack_subprocess_wall_timeout_task_queue",
    "run_multi_attack_subprocess_general",
    "run_multi_attack_subprocess_with_shap",
    "run_multi_attack_subprocess_with_queue",
    "run_multi_attack_subprocess_cpu_timeout",
    "GeneralRunner",
    "QueueRunner",
    "ShapRunner",
    "get_save_dir_from_save_exp",
    "QueueMode",
    "TaskGenerationSpec",
    "GenerationResult",
    "pyct_shap_1_4_8_16_32",
    "pyct_random_1_4_8_16_32",
    "pyct_rnn_random_1_4_8_16_32",
    "pyct_rnn_shap_1_4_8_16_32",
    "stock_shap_1_2_3_4_8_limit_range02",
    "stock_random_1_2_3_4_8_range02",
    "imdb_shap_1_2_3_4_8_range02",
    "imdb_transformer_shap_1_2_3_4_8_range02",
    "mnist_lstm_1_2_3_4_8_range02",
    "mnist_lstm_15_1_2_3_4_8_range02",
    "sentiment_lstm_lstm_15_1_2_3_4_8_range02",
    "fashion_mnist_transformer_shap",
    "fashion_mnist_transformer_shap_calculate_all",
    "fashion_mnist_transformer_random",
]
