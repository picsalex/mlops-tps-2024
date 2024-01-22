from enum import Enum

import hydra
from omegaconf import DictConfig, OmegaConf
from zenml.client import Client
from zenml.enums import ExecutionStatus

from src.pipelines.pipeline_datalake import gitflow_datalake_pipeline
from src.pipelines.pipeline_end_to_end import gitflow_end_to_end_pipeline
from src.pipelines.pipeline_experiment import gitflow_experiment_pipeline


class Pipeline(str, Enum):
    DATALAKE = "datalake"
    EXPERIMENT = "experiment"
    END_TO_END = "end-to-end"


@hydra.main(config_path="src/config/", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    """Main runner for all pipelines.

    Args:
        cfg (DictConfig): Hydra configuration to start the pipeline
    """
    pipeline_name: Pipeline = cfg.pipeline.name

    client = Client()
    orchestrator = client.active_stack.orchestrator
    assert orchestrator is not None, "Orchestrator not in stack."

    if pipeline_name == Pipeline.DATALAKE:
        pipeline_instance = gitflow_datalake_pipeline

    elif pipeline_name == Pipeline.EXPERIMENT:
        pipeline_instance = gitflow_experiment_pipeline

    elif pipeline_name == Pipeline.END_TO_END:
        pipeline_instance = gitflow_end_to_end_pipeline

    else:
        raise ValueError(f"Pipeline name `{pipeline_name}` not supported. ")

    # Run pipeline
    pipeline_instance(cfg=OmegaConf.to_yaml(cfg))

    pipeline_run = pipeline_instance.model.get_runs()[0]

    if pipeline_run.status == ExecutionStatus.FAILED:
        print("Pipeline failed. Check the logs for more details.")
        exit(1)
    elif pipeline_run.status == ExecutionStatus.RUNNING:
        print(
            "Pipeline is still running. The post-execution phase cannot "
            "proceed. Please make sure you use an orchestrator with a "
            "synchronous mode of execution."
        )
        exit(1)


if __name__ == "__main__":
    main()
