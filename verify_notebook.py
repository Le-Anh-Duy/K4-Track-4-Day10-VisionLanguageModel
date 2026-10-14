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
        for yolo in ([.5, .5, 0, .2], [math.nan, .5, .2, .2], [2, .5, .2, .2]):
            with self.assertRaises(ValueError):
                convert("crack", yolo)
        for x in (.1, .5, .9):
            box = convert("pothole", [x, .5, .1234, .2345])["bbox_2d"]
            original = [(x-.1234/2)*640, (.5-.2345/2)*360,
                        (x+.1234/2)*640, (.5+.2345/2)*360]
            actual = [box[0]*.64, box[1]*.36, box[2]*.64, box[3]*.36]
            self.assertTrue(all(abs(a-b) <= .321 for a, b in zip(original, actual)))

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

        env = notebook_functions("is_projector", "save_checkpoint", "load_checkpoint", Path=Path,
                                 processor=SimpleNamespace(save_pretrained=lambda folder: None),
                                 load_base_model=lambda: None,
                                 PeftModel=SimpleNamespace(from_pretrained=lambda *a, **k: TinyModel(0)))
        work = ROOT / "work"
        work.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="notebook-check-", dir=work) as folder:
            env["save_checkpoint"](TinyModel(7), folder, {"train_projector": True}, {"steps": 1})
            restored, run = env["load_checkpoint"](folder)
            self.assertEqual(run["steps"], 1)
            for name, param in restored.named_parameters():
                self.assertTrue((param == (7 if env["is_projector"](name) else 0)).all())
            Path(folder, "extra_trainable.pt").unlink()
            with self.assertRaises(FileNotFoundError):
                env["load_checkpoint"](folder)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
