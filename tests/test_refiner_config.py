"""Configuration checks that do not import PyTorch or load model weights."""
import ast
import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class RefinerConfigTests(unittest.TestCase):
    def test_base_config_matches_dataclass(self):
        config_type = runpy.run_path(str(ROOT / "config_dataclasses.py"))["EncoderRefinerConfig"]
        model = runpy.run_path(str(ROOT / "configs/_base_/model/ovrs_sam3.py"))["model"]
        config = config_type(**model["encoder_refiner_cfg"])
        self.assertEqual(config.local_attn_steps, 4)
        self.assertEqual(config.fusion_layers, 4)
        self.assertNotIn("window_size", config.__dataclass_fields__)
        self.assertNotIn("shift_size", config.__dataclass_fields__)

    def test_refiner_validator_rejects_invalid_steps_and_heads(self):
        config_type = runpy.run_path(str(ROOT / "config_dataclasses.py"))["EncoderRefinerConfig"]
        tree = ast.parse((ROOT / "model_builder.py").read_text())
        method = next(node for node in ast.walk(tree)
                      if isinstance(node, ast.FunctionDef) and node.name == "validate_encoder_refiner_cfg")
        # Execute the production validator in isolation from pretrained-model imports.
        method.decorator_list = []
        namespace = {"EncoderRefinerConfig": config_type}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "model_builder.py", "exec"), namespace)
        coercer = type("Coercer", (), {"_coerce_encoder_refiner_cfg": staticmethod(lambda config: config)})
        validate = lambda **kwargs: namespace[method.name](coercer, config_type(**kwargs))
        self.assertEqual(validate().local_attn_steps, 4)
        for invalid in ({"local_attn_steps": 0}, {"local_attn_steps": -1}, {"num_heads": 3}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate(**invalid)

    def test_configuration_constructor_chain(self):
        paths = ["model_builder.py", "models/sam3_image.py", "models/encoder_refiner.py"]
        classes = {}
        trees = []
        for path in paths + ["models/encoder_refiner_attention.py"]:
            tree = ast.parse((ROOT / path).read_text())
            trees.append(tree)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    constructor = next((item for item in node.body if isinstance(item, ast.FunctionDef)
                                        and item.name == "__init__"), None)
                    if constructor:
                        classes[node.name] = {arg.arg for arg in constructor.args.args}
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in classes:
                    for keyword in node.keywords:
                        if keyword.arg is not None:
                            self.assertIn(keyword.arg, classes[node.func.id], node.func.id)


if __name__ == "__main__":
    unittest.main()
