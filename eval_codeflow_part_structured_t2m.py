"""Run HumanML3D evaluation for Part-Structured CodeFlow checkpoints."""

from models.codeflow import PartStructuredMotionCodeFlow
from models.codeflow.eval_t2m_cli import main


if __name__ == "__main__":
    main(
        model_cls=PartStructuredMotionCodeFlow,
        parser_defaults={
            "output_dir": "./checkpoints/t2m/codeflow_part_structured_pscf",
        },
    )
