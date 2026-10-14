"""Lightweight CPU checks. No model imports, model downloads, CUDA, or dataset needed.

Run: python verify_notebook.py
Tests the actual function definitions extracted from lab_day10.ipynb.
"""
import ast
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace
import unittest
import tempfile
import contextlib
import gc
import shutil
import time
import traceback
from unittest.mock import patch

import torch
from torch.nn.utils.rnn import pad_sequence

from scripts.labcommon import CLASSES
from scripts.score import iou_xyxy, parse_output

ROOT = Path(__file__).resolve().parent
NB = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))


def notebook_functions(*names, **namespace):
    """Execute only requested definitions/constants, never notebook top-level runs."""
    env = dict(torch=torch, json=json, math=math, CLASSES=CLASSES,
               pad_sequence=pad_sequence, parse_output=parse_output, iou_xyxy=iou_xyxy)
    env.update(namespace)
    found = set()
    for cell in NB["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            name = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook>", "exec"), env)
                found.add(name)
    assert found == set(names), (found, names)
    return env


class ToyProcessor:
    """Small deterministic chat fixture, with two visual tokens and a stop token."""
    tokenizer = SimpleNamespace(pad_token_id=0)

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        ids = [11, 99, 99, 12]
        if not add_generation_prompt:
            ids += [100 + ord(c) for c in messages[-1]["content"]] + [13]
        return dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                    mm_token_type_ids=torch.tensor([[0, 1, 1] + [0]*(len(ids)-3)]),
                    image_grid_thw=torch.tensor([[1, 2, 4]]), pixel_values=torch.ones(8, 6))


class NotebookChecks(unittest.TestCase):
    def test_all_cells_compile_and_no_placeholders(self):
        count = 0
        for i, cell in enumerate(NB["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"cell-{i}", "exec")
                self.assertNotIn("YOUR CODE HERE", source)
                count += 1
        self.assertGreaterEqual(count, 10)

    def test_yolo_roundtrip_edges_and_validation(self):
        env = notebook_functions("yolo_to_box", "objects_to_answer")
        convert = env["yolo_to_box"]
        self.assertEqual(convert("crack", [.5, .5, .2, .4])["bbox_2d"], [400, 300, 600, 700])
        self.assertEqual(convert("crack", [1, 1, .0001, .0001])["bbox_2d"], [999, 999, 1000, 1000])
        self.assertEqual(env["objects_to_answer"]([]), "[]")
        for yolo in ([.5, .5, -.1, .2], [math.nan, .5, .2, .2], [2, .5, .2, .2]):
            with self.assertRaises(ValueError):
                convert("crack", yolo)
        for x in (.1, .5, .9):
            box = convert("pothole", [x, .5, .1234, .2345])["bbox_2d"]
            original = [(x-.1234/2)*640, (.5-.2345/2)*360,
                        (x+.1234/2)*640, (.5+.2345/2)*360]
            actual = [box[0]*.64, box[1]*.36, box[2]*.64, box[3]*.36]
            self.assertTrue(all(abs(a-b) <= .321 for a, b in zip(original, actual)))

    def test_zero_area_annotations_do_not_crash_or_become_invented_boxes(self):
        env = notebook_functions("yolo_to_box", "objects_to_answer")
        zero_boxes = [[.5, .5, 0, .2], [.5, .5, .2, 0], [1, 1, 0, 0]]
        for box in zero_boxes:
            self.assertIsNone(env["yolo_to_box"]("crack", box))
        objects = [{"label": "crack", "yolo": box} for box in zero_boxes]
        self.assertEqual(env["objects_to_answer"](objects), "[]")
        objects.append({"label": "pothole", "yolo": [.5, .5, .2, .4]})
        self.assertEqual(json.loads(env["objects_to_answer"](objects)),
                         [{"label": "pothole", "bbox_2d": [400, 300, 600, 700]}])

    def test_mask_and_variable_length_multimodal_collation(self):
        processor = ToyProcessor()
        env = notebook_functions("encode_example", "collate", processor=processor,
                                 user_message=lambda image: {"role": "user", "content": image})
        a = env["encode_example"]("fake image", '[{"label":"crack"}]')
        b = env["encode_example"]("fake image", "[]")
        self.assertTrue((a["labels"][0, :4] == -100).all())
        self.assertTrue(torch.equal(a["labels"][0, 4:], a["input_ids"][0, 4:]))
        self.assertEqual(a["labels"][0, -1].item(), 13)
        batch = env["collate"]([a, b])
        self.assertEqual(batch["pixel_values"].shape, (16, 6))
        self.assertEqual(batch["image_grid_thw"].shape, (2, 3))
        self.assertEqual(batch["mm_token_type_ids"].shape, batch["input_ids"].shape)
        self.assertTrue((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        self.assertTrue((batch["input_ids"][batch["attention_mask"] == 0] == 0).all())
        self.assertIsNone(env["collate"]([None, None]))
        self.assertEqual(env["collate"]([None, b])["input_ids"].shape[0], 1)

    def test_prefix_mismatch_fails_before_wrong_mask(self):
        class BrokenProcessor(ToyProcessor):
            def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
                result = super().apply_chat_template(messages, add_generation_prompt, **kwargs)
                if not add_generation_prompt:
                    result["input_ids"][0, 0] = 77
                return result
        env = notebook_functions("encode_example", processor=BrokenProcessor(), user_message=lambda image: {})
        with self.assertRaises(ValueError):
            env["encode_example"]("fake image", "[]")

    def test_lora_targets_cover_llm_vit_without_projector(self):
        env = notebook_functions("LLM_TARGET", "VIT_TARGET", "is_projector")
        llm = [f"model.language_model.layers.{i}.{suffix}" for i in range(36)
               for suffix in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]
        vit = [f"model.visual.blocks.{i}.{suffix}" for i in range(24)
               for suffix in ["attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2"]]
        self.assertEqual(sum(bool(re.fullmatch(env["LLM_TARGET"], n)) for n in llm+vit), 252)
        combined = f"(?:{env['LLM_TARGET']}|{env['VIT_TARGET']})"
        self.assertEqual(sum(bool(re.fullmatch(combined, n)) for n in llm+vit), 348)
        for name in ("model.visual.merger.linear_fc1.weight", "base_model.model.model.visual.deepstack_merger_list.0.linear_fc1.weight"):
            self.assertTrue(env["is_projector"](name))
            self.assertIsNone(re.fullmatch(combined, name.removesuffix(".weight")))
        self.assertFalse(env["is_projector"]("model.visual.blocks.0.attn.proj.weight"))

    def test_iou_reward_invalid_wrong_class_duplicate_empty(self):
        reward = notebook_functions("iou_reward")["iou_reward"]
        perfect = '[{"label":"pothole","bbox_2d":[100,100,200,200]}]'
        gt = parse_output(perfect)
        self.assertEqual(reward(perfect, gt), 1)
        self.assertEqual(reward("bad json", gt), 0)
        self.assertEqual(reward(perfect.replace("pothole", "crack"), gt), .2)
        self.assertAlmostEqual(reward(json.dumps(json.loads(perfect)*2), gt), .6)
        self.assertEqual(reward("[]", []), 1)
        self.assertEqual(reward("[]", gt), .2)

    def test_projector_checkpoint_roundtrip_with_tiny_tensors(self):
        class TinyModel(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.visual = torch.nn.Module()
                self.model.visual.merger = torch.nn.Linear(2, 2, bias=False)
                self.model.visual.deepstack_merger_list = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])
                self.lora_weight = torch.nn.Parameter(torch.ones(2))
                with torch.no_grad():
                    for p in self.parameters():
                        p.fill_(value)

            def save_pretrained(self, folder):
                (Path(folder) / "adapter_config.json").write_text("{}", encoding="utf-8")

        alerts = []
        env = notebook_functions("is_projector", "write_json_atomic", "remove_checkpoint_dir", "prune_checkpoints",
                                 "save_checkpoint", "load_checkpoint", Path=Path,
                                 processor=SimpleNamespace(save_pretrained=lambda folder: None),
                                 tempfile=tempfile, shutil=shutil, DRIVE_RUN_ROOT=None,
                                 alert=lambda *a, **k: alerts.append((a, k)),
                                 load_base_model=lambda: None,
                                 PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: TinyModel(0)))
        work = ROOT / "work"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="notebook-check-", dir=work) as folder:
            env["DRIVE_RUN_ROOT"] = Path(folder, "unavailable-drive")
            with patch.object(shutil, "copytree", side_effect=OSError("Drive offline")):
                env["save_checkpoint"](TinyModel(7), folder, {"train_projector": True}, {"steps": 1})
            self.assertEqual(len(alerts), 1)
            env["DRIVE_RUN_ROOT"] = None
            restored, run = env["load_checkpoint"](folder)
            self.assertEqual(run["steps"], 1)
            for name, param in restored.named_parameters():
                self.assertTrue((param == (7 if env["is_projector"](name) else 0)).all())
            state = json.loads(Path(folder, "run.json").read_text(encoding="utf-8"))
            # A failed subsequent save must leave the last complete checkpoint usable.
            env["processor"] = SimpleNamespace(save_pretrained=lambda p: (_ for _ in ()).throw(OSError("disk full")))
            with self.assertRaises(OSError):
                env["save_checkpoint"](TinyModel(9), folder, {"train_projector": True}, {"steps": 2})
            self.assertEqual(json.loads(Path(folder, "run.json").read_text(encoding="utf-8")), state)
            restored, _ = env["load_checkpoint"](folder)
            self.assertTrue((restored.model.visual.merger.weight == 7).all())
            Path(folder, state["checkpoint"], "extra_trainable.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                env["load_checkpoint"](folder)

    def test_checkpoint_pruning_keeps_current_when_step_number_restarts(self):
        env = notebook_functions("remove_checkpoint_dir", "prune_checkpoints", Path=Path,
                                 shutil=shutil, alert=lambda *a, **k: None)
        with tempfile.TemporaryDirectory(prefix="notebook-prune-", dir=ROOT / "work") as folder:
            root = Path(folder)
            for name in ("checkpoint-000300-old", "checkpoint-000250-old", "checkpoint-000010-new"):
                (root / name).mkdir()
            current = root / "checkpoint-000010-new"
            env["prune_checkpoints"](root, current)
            self.assertTrue(current.is_dir())
            self.assertEqual(len(list(root.glob("checkpoint-*"))), 2)
            with self.assertRaises(ValueError):
                env["remove_checkpoint_dir"](root / "checkpoint-000010-new", root)

    def test_safe_task_logs_failure_and_next_task_still_runs(self):
        alerts = []
        env = notebook_functions("safe_task", alert=lambda *a, **k: alerts.append((a, k)),
                                 traceback=traceback, gc=gc)
        self.assertIsNone(env["safe_task"]("broken", lambda: 1/0))
        self.assertEqual(env["safe_task"]("next", lambda: 42), 42)
        self.assertEqual(len(alerts), 1)

    def test_e4_without_successful_baseline_alerts_and_returns(self):
        alerts = []
        env = notebook_functions("run_e4", results={"E1": None, "E2": {"val": None, "test": None}},
                                 alert=lambda *a, **k: alerts.append(a))
        self.assertIsNone(env["run_e4"]())
        self.assertEqual(len(alerts), 1)

    def test_failed_validation_inference_does_not_block_test_split(self):
        alerts, trained = [], []

        def predict(model, rows, path):
            if rows == ["bad"]:
                raise OSError("simulated validation failure")
            return path

        with tempfile.TemporaryDirectory(prefix="notebook-run-", dir=ROOT / "work") as folder:
            root = Path(folder)
            env = notebook_functions("safe_task", "run_experiment", Path=Path, gc=gc, traceback=traceback,
                alert=lambda *a, **k: alerts.append((a, k)), ROOT=root, RUN_ROOT=root / "runs",
                DRIVE_RUN_ROOT=None, REUSE_COMPLETED=True, dataset_version="test-only",
                splits={"val": ["bad"], "test": ["good"]}, predict=predict,
                train_run=lambda *a: trained.append(1),
                load_checkpoint=lambda folder: (object(), dict(trainable_params=2, peak_vram_gb=.001, train_minutes=.001)),
                score_run=lambda *a, **k: {"split": k["split"]})
            result = env["run_experiment"]({"name": "E1", "max_steps": 2}, [])
            self.assertEqual(trained, [1])
            self.assertIsNone(result["val"])
            self.assertEqual(result["test"], {"split": "test"})
            self.assertEqual(len(alerts), 1)

    def exercise_tiny_training(self, modes, max_steps=2):
        class BadGradient(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, grad):
                return torch.full_like(grad, float("inf"))

        class TinyTrainModel(torch.nn.Module):
            device = torch.device("cpu")

            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.25))
                self.calls = 0

            def forward(self, **batch):
                mode = modes[min(self.calls, len(modes)-1)]
                self.calls += 1
                if mode == "oom":
                    raise torch.cuda.OutOfMemoryError("simulated OOM; no GPU used")
                if mode == "nan":
                    loss = self.weight * float("nan")
                elif mode == "inf_grad":
                    loss = BadGradient.apply(self.weight)
                else:
                    loss = (self.weight-batch["input_ids"].mean()).square()
                return SimpleNamespace(loss=loss)

        class Progress:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, amount):
                pass

        alerts, checkpoints, scheduler_steps, optimizer_steps = [], [], [], []
        original_optimizer = torch.optim.AdamW
        original_scaler = torch.amp.GradScaler

        class RecordingOptimizer(original_optimizer):
            def step(self, *args, **kwargs):
                optimizer_steps.append(1)
                return super().step(*args, **kwargs)

        model = TinyTrainModel()
        from torch.utils.data import DataLoader
        env = notebook_functions("train_run", set_seed=lambda seed: None, load_base_model=lambda: model,
             apply_setup=lambda base, cfg: base, RoadDataset=lambda rows: rows, DataLoader=DataLoader,
             collate=lambda rows: {"input_ids": torch.stack(rows)}, gc=gc, time=time,
             tqdm=lambda **kwargs: Progress(), alert=lambda *a, **k: alerts.append((a, k)),
             get_cosine_schedule_with_warmup=lambda *a, **k: SimpleNamespace(step=lambda: scheduler_steps.append(1)),
             checkpoint_safely=lambda model, folder, cfg, stats, **kwargs: checkpoints.append(dict(stats)) or True)
        cfg = dict(name="tiny", seed=42, batch_size=1, grad_accum=1, lr=.01, max_steps=max_steps)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(torch, "autocast", lambda *a, **k: contextlib.nullcontext()))
            stack.enter_context(patch.object(torch.optim, "AdamW", RecordingOptimizer))
            stack.enter_context(patch.object(torch.amp, "GradScaler", lambda *a, **k: original_scaler("cpu", **k)))
            for name in ("reset_peak_memory_stats", "synchronize", "empty_cache"):
                stack.enter_context(patch.object(torch.cuda, name, lambda: None))
            stack.enter_context(patch.object(torch.cuda, "max_memory_allocated", lambda: 1024))
            result = env["train_run"](cfg, [torch.tensor(1.)]*4, "unused-test-folder")
        return result, checkpoints, optimizer_steps, scheduler_steps, alerts

    def test_training_skips_oom_nan_and_inf_grad_without_counting_optimizer_steps(self):
        result, checkpoints, optimizer, scheduler, alerts = self.exercise_tiny_training(
            ["oom", "nan", "inf_grad", "good"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["rejected_updates"], 3)
        self.assertEqual(len(optimizer), 2)
        self.assertEqual(len(scheduler), 2)
        self.assertEqual(checkpoints[-1]["steps"], 2)
        self.assertEqual(len(alerts), 3)

    def test_training_does_not_retry_bad_batches_forever(self):
        with self.assertRaisesRegex(RuntimeError, "Quá nhiều"):
            self.exercise_tiny_training(["nan"])

    def test_inference_splits_oom_batch_and_records_unreadable_image(self):
        class Batch(dict):
            def to(self, device):
                return self

        class Processor:
            tokenizer = SimpleNamespace(padding_side="left")

            def apply_chat_template(self, chats, **kwargs):
                return Batch(input_ids=torch.tensor([[chat[0]["image"], 9] for chat in chats]))

            def batch_decode(self, ids, **kwargs):
                return [str(row[0].item()) for row in ids]

        class Model:
            config = SimpleNamespace(use_cache=False)
            device = torch.device("cpu")

            def eval(self):
                pass

            def generate(self, input_ids, **kwargs):
                if len(input_ids) > 1:
                    raise torch.cuda.OutOfMemoryError("simulated batch OOM")
                return torch.cat([input_ids, input_ids[:, :1]], dim=1)

        def open_image(row):
            if row["image_id"] == 2:
                raise OSError("simulated broken image")
            return row["image_id"]

        alerts = []
        env = notebook_functions("predict", Path=Path, gc=gc, processor=Processor(),
                                 user_message=lambda image: {"image": image}, open_image=open_image,
                                 tqdm=lambda iterable, **kwargs: iterable,
                                 alert=lambda *a, **k: alerts.append((a, k)))
        with tempfile.TemporaryDirectory(prefix="notebook-infer-", dir=ROOT / "work") as folder:
            path = Path(folder, "pred.jsonl")
            with patch.object(torch, "autocast", lambda *a, **k: contextlib.nullcontext()):
                env["predict"](Model(), [{"image_id": i} for i in range(4)], path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["raw"] for row in rows], ["0", "1", "", "3"])
            self.assertEqual([row["image_id"] for row in rows], [0, 1, 2, 3])
            self.assertFalse(path.with_suffix(".jsonl.tmp").exists())
            self.assertTrue(alerts)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
