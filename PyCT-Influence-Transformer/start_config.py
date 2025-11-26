_INPUT_PREFIX = "fashion_mnist_test_"
_QUEUE_TYPE = "priority_queue"
_LOG_LEVEL_CHOICES = ("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG")
_DEFAULT_PIXEL_SEARCH = (1, 2, 4, 8, 16, 32)

# Re-export for convenience
__all__ = [
    "_INPUT_PREFIX",
    "_QUEUE_TYPE",
    "_LOG_LEVEL_CHOICES",
    "_DEFAULT_PIXEL_SEARCH",
]
