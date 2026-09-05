"""Offline orchestration checks; model subprocesses and metrics are test fixtures."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from mecta.evaluation import read_predictions, write_predictions
from mecta.io import iter_jsonl, read_json, write_json, write_jsonl
from mecta.pipeline import STAGE_NAMES, artifact_paths, select_stages
from mecta.schema import Article, Constraint, TimelineEvent
from mecta.stage1_generation import generate_partition
from scripts import run_pipeline


class TestFullPipeline(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = self.root / "experiment.yaml"
        self.config.write_text("dataset: crest\n", encoding="utf-8")
        self.run_dir = self.root / "run"
        self.paths = artifact_paths(self.run_dir)
        self.commands: list[list[str]] = []
        self.selection = {"checkpoint": "fixture", "threshold": 0.5}
        self.predictions = {
            ("entity", "0"): (
                TimelineEvent(
                    "entity", "0", "candidate", "2020-01-02", "Generated event", 0.8
                ),
            )
        }
        config_loader = patch.object(run_pipeline, "load_config", return_value={})
        config_loader.start()
        self.addCleanup(config_loader.stop)

    def _fixture_command(self, command: list[str], log_path: Path) -> None:
        """Write minimal outputs while checking each command consumes its inputs."""
        self.commands.append(list(command))
        script = Path(command[1]).name
        options = dict(zip(command[2::2], command[3::2]))
        self.assertEqual(Path(options["--config"]), self.config.resolve())
        self.assertEqual(log_path.parent, self.paths["logs"])

        def require_file(path: str | Path) -> None:
            self.assertTrue(Path(path).is_file(), f"{script} needs {path}")

        if script == "prepare_stage1.py":
            write_jsonl(Path(options["--output-dir"]) / "train.jsonl", [])
        elif script == "train_stage1.py":
            require_file(options["--train-file"])
            write_json(
                Path(options["--output-dir"]) / "final_adapter" / "adapter_config.json",
                {"fixture": True},
            )
        elif script == "generate_mentions.py":
            require_file(Path(options["--adapter"]) / "adapter_config.json")
            write_json(
                Path(options["--output-dir"]) / "_meta" / "summary.json",
                {"partition": options["--partition"]},
            )
        elif script == "build_candidates.py":
            require_file(Path(options["--mentions-dir"]) / "_meta" / "summary.json")
            write_json(Path(options["--output-dir"]) / "_meta" / "summary.json", {})
        elif script == "prepare_stage2.py":
            self.assertEqual(options["--partition"], "train")
            require_file(Path(options["--candidates-dir"]) / "_meta" / "summary.json")
            write_jsonl(options["--output"], [])
        elif script == "train_stage2.py":
            require_file(options["--train-pairs"])
            require_file(
                Path(options["--development-candidates-dir"]) / "_meta" / "summary.json"
            )
            write_json(Path(options["--output-dir"]) / "selected_config.json", self.selection)
        elif script == "build_timelines.py":
            self.assertEqual(options["--partition"], "test")
            self.assertEqual(read_json(options["--selection"]), self.selection)
            require_file(Path(options["--candidates-dir"]) / "_meta" / "summary.json")
            write_predictions(options["--output"], self.predictions)
        elif script == "evaluate.py":
            self.assertEqual(options["--partition"], "test")
            self.assertEqual(read_predictions(options["--predictions"]), self.predictions)
            write_json(options["--output"], {"fixture_only": True, "timeline_count": 1})
        else:
            self.fail(f"Unexpected pipeline command: {script}")

    def _run(self, *arguments: str) -> str:
        output = io.StringIO()
        with patch.object(run_pipeline, "_run_command", side_effect=self._fixture_command):
            with redirect_stdout(output):
                status = run_pipeline.main(
                    [
                        "--config", str(self.config),
                        "--run-dir", str(self.run_dir),
                        "--device", "cpu",
                        *arguments,
                    ]
                )
        self.assertEqual(status, 0)
        return output.getvalue()

    def test_default_runs_all_stages_and_uses_trained_adapter_for_every_partition(self) -> None:
        self.assertEqual(select_stages(), STAGE_NAMES)
        output = self._run()
        started = tuple(line[6:] for line in output.splitlines() if line.startswith("START "))
        self.assertEqual(started, STAGE_NAMES)
        scripts = [Path(command[1]).name for command in self.commands]
        self.assertEqual(scripts[:2], ["prepare_stage1.py", "train_stage1.py"])
        generation = [
            command for command in self.commands
            if Path(command[1]).name == "generate_mentions.py"
        ]
        self.assertEqual(
            [command[command.index("--partition") + 1] for command in generation],
            ["train", "development", "test"],
        )
        training = self.commands[1]
        adapter = Path(training[training.index("--output-dir") + 1]) / "final_adapter"
        for command in generation:
            self.assertEqual(Path(command[command.index("--adapter") + 1]), adapter)
        self.assertEqual(scripts[-1], "evaluate.py")
        self.assertEqual(read_json(self.paths["selection"]), self.selection)
        self.assertEqual(
            read_predictions(self.paths["timelines"] / "test_predictions.jsonl"),
            self.predictions,
        )
        crest_timeline = self.paths["timelines"] / "crest" / "entity" / "0" / "timelines.jsonl"
        self.assertEqual(
            read_json(crest_timeline),
            [["2020-01-02 00:00:00", ["Generated event"]]],
        )
        self.assertEqual(
            read_json(self.paths["evaluation"] / "test_metrics.json"),
            {"fixture_only": True, "timeline_count": 1},
        )
        self.assertEqual(
            read_json(self.run_dir / "run_manifest.json"), run_pipeline.RUN_MANIFEST
        )

    def test_resume_finished_run_skips_all_stages(self) -> None:
        self._run()
        self.commands.clear()
        output = self._run("--resume")
        self.assertEqual(self.commands, [])
        self.assertEqual(
            [
                line.split(":", 1)[0][5:] for line in output.splitlines()
                if line.startswith("SKIP ")
            ],
            list(STAGE_NAMES),
        )

    def test_resume_after_stage1_continues_with_generation(self) -> None:
        self._run("--stop-after", "train_stage1")
        self.commands.clear()
        self._run("--resume")
        self.assertEqual(Path(self.commands[0][1]).name, "generate_mentions.py")
        self.assertFalse(any(
            Path(command[1]).name == "train_stage1.py" for command in self.commands
        ))
        self.assertTrue((self.paths["evaluation"] / "test_metrics.json").is_file())

    def test_resume_rejects_unverified_runs_before_writing_or_dispatching(self) -> None:
        for index, manifest in enumerate((None, "{}", '{"workflow": "other"}', "invalid JSON")):
            with self.subTest(manifest=manifest):
                run_dir = self.root / f"unverified_{index}"
                write_json(run_dir / "mentions" / "test" / "_meta" / "summary.json", {})
                if manifest is not None:
                    (run_dir / "run_manifest.json").write_text(manifest, encoding="utf-8")
                before = {
                    path.relative_to(run_dir): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                with patch.object(run_pipeline, "_run_command") as dispatch:
                    with self.assertRaises(ValueError):
                        run_pipeline.main(
                            [
                                "--config", str(self.config), "--run-dir", str(run_dir),
                                "--resume", "--from-stage", "generate_test",
                            ]
                        )
                dispatch.assert_not_called()
                after = {
                    path.relative_to(run_dir): path.read_bytes()
                    for path in run_dir.rglob("*") if path.is_file()
                }
                self.assertEqual(after, before)

    def test_missing_stage_output_aborts_before_downstream_work(self) -> None:
        with patch.object(run_pipeline, "_run_command") as dispatch:
            with redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "prepare_stage1"):
                    run_pipeline.main(["--config", str(self.config), "--run-dir", str(self.run_dir)])
        self.assertEqual(dispatch.call_count, 1)
        self.assertFalse(self.paths["mentions"].exists())

    def test_generation_reads_articles_and_constraints_without_reading_labels(self) -> None:
        article = Article(
            "entity", "article", "2020-01-01", "Article title",
            "Complete article body and its final sentence.",
        )
        constraints = tuple(
            Constraint("entity", str(index), f"Constraint {index}") for index in range(5)
        )
        reader = Mock(spec=[
            "entity_ids", "constraints_for", "articles_for", "references", "references_for"
        ])
        reader.entity_ids.return_value = ("entity",)
        reader.constraints_for.return_value = {"entity": constraints}
        reader.articles_for.return_value = {"entity": (article,)}
        reader.references.side_effect = AssertionError("Generation must not read labels")
        reader.references_for.side_effect = AssertionError("Generation must not read labels")
        generator = Mock(spec=["generate_batch"])
        generator.generate_batch.return_value = (
            '[{"date":"2020-01-02","event_summary":"Generated event"}]',
        )
        for partition in ("train", "development", "test"):
            with self.subTest(partition=partition):
                destination = self.root / f"generated_{partition}"
                summary = generate_partition(
                    reader, partition, generator, destination,
                    {"require_explicit_target_name": False}, seed=42,
                )
                reader.entity_ids.assert_called_with(partition)
                self.assertEqual(summary["mentions"], 1)
                messages = generator.generate_batch.call_args.args[0][0]
                self.assertEqual(
                    [message["role"] for message in messages], ["system", "user"]
                )
                prompt = messages[1]["content"]
                self.assertIn(article.title, prompt)
                self.assertIn(article.text, prompt)
                for constraint in constraints:
                    self.assertIn(constraint.text, prompt)
                mention, = iter_jsonl(destination / "entity.jsonl")
                self.assertEqual(mention["article_id"], article.article_id)
                self.assertEqual(mention["summary"], "Generated event")
        reader.references.assert_not_called()
        reader.references_for.assert_not_called()


if __name__ == "__main__":
    unittest.main()
