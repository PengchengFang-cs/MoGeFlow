"""Run HumanML3D evaluation for Part-Structured CodeFlow checkpoints."""

from models.codeflow import PartStructuredMotionCodeFlow
from eval_codeflow_t2m import main


if __name__ == "__main__":
    main(
        model_cls=PartStructuredMotionCodeFlow,
        parser_defaults={
            "output_dir": "./checkpoints/t2m/codeflow_part_structured_pscf",
        },
    )
