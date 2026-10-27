"""Literal operands carried by source-projected arithmetic nodes."""
import torch


def apply_scalar(op_key, inputs, params):
    side = params.get("_scalar_side")
    if side not in {"left", "right"}:
        return None
    op = op_key.removeprefix("torch.")
    if op not in {"add", "sub", "mul", "div"} or len(inputs) != 1 or type(params.get("other")) not in (int, float):
        raise ValueError("Scalar arithmetic requires one tensor and a numeric literal operand")
    left, right = (params["other"], inputs[0]) if side == "left" else (inputs[0], params["other"])
    return getattr(torch, op)(left, right)
