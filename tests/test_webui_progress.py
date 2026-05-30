from __future__ import annotations

import unittest

from webui.progress import Evaluation, LossSample, TrainingProgress, parse_training_progress


class ProgressTest(unittest.TestCase):
    def test_parses_full_tqdm_line(self) -> None:
        line = (
            "Training progress:  45%|████▌     | 13500/30000 "
            "[02:14<02:43,  100.50it/s, Loss=0.0312000, Depth Loss=0.0000000]"
        )
        progress = parse_training_progress(line)
        self.assertTrue(progress.has_data)
        self.assertEqual(progress.iteration, 13500)
        self.assertEqual(progress.total_iterations, 30000)
        self.assertAlmostEqual(progress.percent, 45.0)
        self.assertAlmostEqual(progress.rate, 100.5)
        self.assertEqual(progress.elapsed, "02:14")
        self.assertEqual(progress.eta, "02:43")
        self.assertAlmostEqual(progress.loss, 0.0312)
        self.assertAlmostEqual(progress.depth_loss, 0.0)

    def test_tqdm_line_without_postfix_and_unknown_eta(self) -> None:
        line = "Training progress:   0%|          | 0/30000 [00:00<?, ?it/s]"
        progress = parse_training_progress(line)
        self.assertTrue(progress.has_data)
        self.assertEqual(progress.iteration, 0)
        self.assertEqual(progress.total_iterations, 30000)
        self.assertEqual(progress.eta, "?")
        self.assertIsNone(progress.rate)
        self.assertIsNone(progress.loss)
        self.assertIsNone(progress.depth_loss)

    def test_collects_evaluations(self) -> None:
        text = "\n".join(
            [
                "[ITER 7000] Evaluating test: L1 0.034 PSNR 28.51",
                "[ITER 7000] Evaluating train: L1 0.012 PSNR 31.20",
                "[ITER 30000] Evaluating test: L1 0.021 PSNR 30.10",
                "[ITER 30000] Evaluating train: L1 0.008 PSNR 33.40",
            ]
        )
        progress = parse_training_progress(text)
        self.assertEqual(len(progress.evaluations), 4)
        self.assertEqual(progress.evaluations[0], Evaluation(7000, "test", 0.034, 28.51))
        self.assertEqual(progress.evaluations[1].split, "train")
        self.assertEqual(progress.evaluations[3].iteration, 30000)
        tests = [e for e in progress.evaluations if e.split == "test"]
        self.assertEqual(len(tests), 2)

    def test_scientific_notation_eval(self) -> None:
        progress = parse_training_progress(
            "[ITER 100] Evaluating test: L1 1.2e-03 PSNR 9.5"
        )
        self.assertEqual(len(progress.evaluations), 1)
        self.assertAlmostEqual(progress.evaluations[0].l1, 0.0012)

    def test_saved_iterations_dedup_in_order(self) -> None:
        text = "\n".join(
            [
                "[ITER 7000] Saving Gaussians",
                "[ITER 30000] Saving Gaussians",
                "[ITER 7000] Saving Gaussians",
            ]
        )
        progress = parse_training_progress(text)
        self.assertEqual(progress.saved_iterations, [7000, 30000])

    def test_output_folder(self) -> None:
        progress = parse_training_progress("Output folder: output/foo\n")
        self.assertEqual(progress.output_folder, "output/foo")
        self.assertTrue(progress.has_data)

    def test_empty_and_unrelated_text(self) -> None:
        for text in ["", "some colmap line\nanother line"]:
            progress = parse_training_progress(text)
            self.assertIsInstance(progress, TrainingProgress)
            self.assertFalse(progress.has_data)
            self.assertEqual(progress.evaluations, [])
            self.assertIsNone(progress.iteration)

    def test_last_tqdm_line_wins(self) -> None:
        text = "\n".join(
            [
                "Training progress:  10%|█   | 3000/30000 [00:30<04:30, 100.00it/s, Loss=0.1, Depth Loss=0.0]",
                "Training progress:  20%|██  | 6000/30000 [01:00<04:00, 100.00it/s, Loss=0.05, Depth Loss=0.0]",
            ]
        )
        progress = parse_training_progress(text)
        self.assertEqual(progress.iteration, 6000)
        self.assertAlmostEqual(progress.loss, 0.05)
        self.assertEqual(
            progress.loss_samples,
            [LossSample(3000, 0.1, 0.0), LossSample(6000, 0.05, 0.0)],
        )


if __name__ == "__main__":
    unittest.main()
