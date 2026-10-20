"""Exercise submission boundaries using a fake scheduler, never a real cluster."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = (r"C:\Program Files\Git\bin\bash.exe" if os.name == "nt" else shutil.which("bash"))


def bash_path(path):
    path = Path(path).resolve()
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


@unittest.skipUnless(BASH and Path(BASH).exists(), "Bash required")
class SubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "repo/.git").mkdir(parents=True)
        (self.root / "data/datasets").mkdir(parents=True)
        (self.root / "data/datasets/pusht_expert_train.h5").touch()
        (self.root / "bin").mkdir()
        scheduler = self.root / "bin/sbatch"
        scheduler.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nprintf "12345\\n"\n')
        scheduler.chmod(0o755)
        self.env = dict(os.environ)
        self.env.update({
            "FLOWJEPA_REPO": bash_path(self.root / "repo"),
            "FLOWJEPA_PYTHON": bash_path(sys.executable),
            "FLOWJEPA_DATA_ROOT": bash_path(self.root / "data"),
            "FLOWJEPA_RUNS_ROOT": bash_path(self.root / "runs"),
            "FLOWJEPA_LOG_ROOT": bash_path(self.root / "logs"),
            "FLOWJEPA_RUN_ID": "flowjepa-boundary-test",
            "FLOWJEPA_PARTITION": "candidate", "FLOWJEPA_QOS": "candidate",
            "FLOWJEPA_TIME": "00:15:00", "CAPTURE": bash_path(self.root / "capture"),
            "FLOWJEPA_WANDB": "0",
            "PATH": str(self.root / "bin") + os.pathsep + os.environ["PATH"],
        })

    def run_script(self, mode="train", action="--preview"):
        return subprocess.run([BASH, bash_path(ROOT / "scripts/circe_submit.sh"), mode, action],
                              env=self.env, capture_output=True, text=True)

    def test_preview_does_not_contact_scheduler(self):
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root / "capture").exists())
        self.assertFalse((self.root / "logs").exists())

    def test_scheduler_check_has_absolute_paths_and_no_submission(self):
        result = self.run_script(action="--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        args = (self.root / "capture").read_text().splitlines()
        self.assertIn("--test-only", args)
        self.assertIn("--chdir=" + self.env["FLOWJEPA_REPO"], args)
        self.assertIn("--gres=gpu:1", args)
        self.assertIn("--cpus-per-task=8", args)
        self.assertIn("--mem=64G", args)
        self.assertTrue(any(a.startswith("--output=" + self.env["FLOWJEPA_LOG_ROOT"]) for a in args))

    def test_existing_experiment_rejects_fresh_training(self):
        (self.root / "runs/flowjepa-boundary-test").mkdir(parents=True)
        self.assertNotEqual(self.run_script().returncode, 0)
        self.assertFalse((self.root / "capture").exists())

    def test_foreign_checkpoint_is_rejected(self):
        checkpoint = self.root / "foreign/last.ckpt"
        checkpoint.parent.mkdir()
        checkpoint.touch()
        self.env["FLOWJEPA_RESUME"] = bash_path(checkpoint)
        self.assertNotEqual(self.run_script(mode="resume").returncode, 0)
        self.assertFalse((self.root / "capture").exists())

    def test_own_checkpoint_is_accepted_for_resume_preview(self):
        checkpoint = self.root / "runs/flowjepa-boundary-test/trainer/runs/test/checkpoints/last.ckpt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        self.env["FLOWJEPA_RESUME"] = bash_path(checkpoint)
        result = self.run_script(mode="resume")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("checkpoint.resume=", result.stderr)
        self.assertFalse((self.root / "capture").exists())

    def test_explicit_submit_returns_only_job_id_on_stdout(self):
        result = self.run_script(action="--submit")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "12345")
        self.assertNotIn("--test-only", (self.root / "capture").read_text().splitlines())


if __name__ == "__main__":
    unittest.main()
