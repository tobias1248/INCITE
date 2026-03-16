#!/usr/bin/env python3
"""Tests for compact position logging helpers."""

from __future__ import annotations

import unittest

from libct.position import summarize_indices, summarize_position


class PositionLoggingTests(unittest.TestCase):
    def test_summarize_indices_short_list(self) -> None:
        indices = [(1, 0), (1, 1)]
        text = summarize_indices(indices)
        self.assertIn("(len=2)", text)
        self.assertIn("(1, 0)", text)

    def test_summarize_indices_long_list(self) -> None:
        indices = [(2, i) for i in range(10)]
        text = summarize_indices(indices, preview=2)
        self.assertIn("(len=10)", text)
        self.assertIn("...", text)
        self.assertIn("(2, 0)", text)
        self.assertIn("(2, 9)", text)

    def test_summarize_position(self) -> None:
        position = (12, [(2, i) for i in range(8)])
        text = summarize_position(position, preview=2)
        self.assertIn("layer=12", text)
        self.assertIn("(len=8)", text)


if __name__ == "__main__":
    unittest.main()
