"""Lightweight checks for the imported submission; never loads torch or a model.

Run: python verify_notebook.py
"""
import ast
import json
from pathlib import Path
import re
import unittest

from PIL import Image
from scripts.score import iou_xyxy, parse_output

ROOT = Path(__file__).resolve().parent
NB = json.loads((ROOT / "lab_day10.ipynb").read_text(encoding="utf-8"))
REPORT = (ROOT / "REPORT.md").read_text(encoding="utf-8")
EVIDENCE = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in (ROOT / "report/evidence").glob("*.json")}


def definitions(*names):
    """Execute only selected pure definitions/constants, not notebook cells."""
    env = dict(json=json, re=re, parse_output=parse_output, iou_xyxy=iou_xyxy)
    found = set()
    for cell in NB["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            name = node.name if isinstance(node, ast.FunctionDef) else None
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
            if name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<notebook>", "exec"), env)
                found.add(name)
    assert found == set(names), (found, set(names))
    return env


class SubmissionChecks(unittest.TestCase):
    def test_notebook_syntax_sections_and_identity(self):
        self.assertEqual(NB["nbformat"], 4)
        headings = []
        for i, cell in enumerate(NB["cells"]):
            self.assertIsInstance(cell["source"], list)
            self.assertTrue(all(line.endswith("\n") for line in cell["source"][:-1]))
            text = "".join(cell["source"])
            if cell["cell_type"] == "code":
                compile(text, f"cell-{i}", "exec")
            else:
                headings.append(text.splitlines()[0])
        for i in range(9):
            self.assertTrue(any(h.startswith(f"## {i}.") for h in headings))
        for text in (REPORT, "".join(NB["cells"][0]["source"])):
            self.assertIn("Lê Anh Duy", text)
            self.assertIn("2A202602723", text)
            self.assertEqual(set(re.findall(r"\b2A\d{9}\b", text)), {"2A202602723"})
        self.assertNotIn("<điền", REPORT)
        self.assertEqual(len(re.findall(r"^### Câu \d+", REPORT, re.M)), 10)
        self.assertIn("github.com/Le-Anh-Duy/", "".join(NB["cells"][2]["source"]))

    def test_evidence_arithmetic_and_dataset_consistency(self):
        required = {"e0", "e1", "e2", "e3", "e4_coord_norm1000", "e4_coord_pixel"}
        self.assertTrue(required <= EVIDENCE.keys())
        self.assertEqual({r["dataset_version"] for r in EVIDENCE.values()}, {"d10-21e4070301"})
        for name, result in EVIDENCE.items():
            with self.subTest(name=name):
                self.assertEqual(result["score_version"], 1)
                self.assertEqual(result["split"], "test")
                self.assertEqual(result["n_images"], 311)
                self.assertEqual(result["missing_predictions"], 0)
                micro = result["micro"]
                tp, fp, fn = (micro[k] for k in ("tp", "fp", "fn"))
                self.assertAlmostEqual(micro["precision"], tp/(tp+fp))
                self.assertAlmostEqual(micro["recall"], tp/(tp+fn))
                self.assertAlmostEqual(micro["f1"], 2*tp/(2*tp+fp+fn))
                self.assertAlmostEqual(result["fp_per_image"], fp/311)
                for key in ("tp", "fp", "fn"):
                    self.assertEqual(micro[key], sum(v[key] for v in result["per_class"].values()))
                self.assertEqual([result["per_class"][c]["tp"]+result["per_class"][c]["fn"]
                                  for c in ("pothole", "crack", "manhole")], [121, 299, 163])
                self.assertAlmostEqual(result["parse_valid_rate"]*311,
                                       round(result["parse_valid_rate"]*311))
                self.assertLessEqual(result["f1_ci95"][0], micro["f1"])
                self.assertGreaterEqual(result["f1_ci95"][1], micro["f1"])

    def test_report_table_matches_evidence(self):
        rows = [line for line in REPORT.splitlines() if line.startswith("| E")]
        tags = ["e0", "e1", "e2", "e3", "e4_coord_norm1000", "e4_coord_pixel"]
        self.assertEqual(len(rows), len(tags))
        for row, tag in zip(rows, tags):
            values = [v.strip().replace("**", "") for v in row.strip("|").split("|")]
            r = EVIDENCE[tag]
            self.assertEqual(values[5], f"{r['micro']['f1']:.4f}")
            self.assertEqual(values[6], f"{r['f1_ci95'][0]:.4f}–{r['f1_ci95'][1]:.4f}")
            self.assertEqual(values[7], f"{r['fp_per_image']:.3f}")
            for index, label in enumerate(("pothole", "crack", "manhole"), 8):
                self.assertEqual(values[index], f"{r['per_class'][label]['f1']:.4f}")
            if tag != "e0":
                self.assertEqual(values[2], f"{r['meta']['trainable_params']:,}")
                self.assertEqual(values[3], f"{r['meta']['peak_vram_gb']:.3f}")
                self.assertEqual(values[4], f"{r['meta']['train_minutes']:.2f}")
                self.assertEqual(r["meta"]["max_steps"], 178)
        self.assertEqual(round(EVIDENCE["e0"]["parse_valid_rate"]*311), 228)

    def test_e4_baseline_and_limits_are_explicit(self):
        for key in ("micro", "per_class", "meta", "f1_ci95"):
            self.assertEqual(EVIDENCE["e2"][key], EVIDENCE["e4_coord_norm1000"][key])
        self.assertEqual(EVIDENCE["e4_coord_pixel"]["coord"], "pixel")
        self.assertEqual(EVIDENCE["e4_coord_pixel"]["meta"]["setup"], "E2")
        env = definitions("E4_BASE_SETUP")
        self.assertEqual(env["E4_BASE_SETUP"], "E2")
        self.assertIn("178", REPORT)
        self.assertIn("300", REPORT)
        self.assertIn("không chứa prediction", REPORT)

    def test_pure_label_conversion_and_reward(self):
        env = definitions("yolo_to_box", "objects_to_answer", "objects_to_answer_px", "IOU_THR", "reward_iou")
        obj = {"label": "crack", "yolo": [.5, .5, .2, .4]}
        self.assertEqual(env["yolo_to_box"](**obj)["bbox_2d"], [400, 300, 600, 700])
        self.assertEqual(env["objects_to_answer"]([]), "[]")
        self.assertEqual(json.loads(env["objects_to_answer_px"]([obj]))[0]["bbox_2d"], [256, 108, 384, 252])
        gt = [{"label": "pothole", "bbox": (0, 0, 100, 100)}]
        perfect = '[{"label":"pothole","bbox_2d":[0,0,100,100]}]'
        reward = env["reward_iou"]
        for raw, gts, expected in [(perfect, gt, 1), ("bad json", gt, 0),
                (perfect.replace("pothole", "crack"), gt, .2), ("[]", gt, .2), ("[]", [], 1),
                (json.dumps(json.loads(perfect)*2), gt, .6)]:
            self.assertAlmostEqual(reward([raw], gt_boxes=[gts])[0], expected)

    def test_report_image_links_exist_and_pngs_are_readable(self):
        links = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", REPORT)
        self.assertGreaterEqual(len(links), 2)
        for link in links:
            self.assertTrue((ROOT / link).is_file(), link)
        for path in (ROOT / "report/figures").glob("*.png"):
            with Image.open(path) as image:
                image.verify()


if __name__ == "__main__":
    unittest.main(verbosity=2)
