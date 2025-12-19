
# import argparse
import json
from sagemaker.workflow.pipeline import Pipeline


def main():  # pragma: no cover
    
    with open("train_config.json", "r") as f:
        training_config = json.load(f)
    print(training_config["PipelineName"])
    pipeline = Pipeline(
        name=training_config["PipelineName"])
    
    pipeline.start(
        parameters=dict(
            AlgorithmMetric="auc",
            TunerObjectiveMetric = "auc",
            TunerMetricDefinition="auc: ([0-9\\.]+)",
            OptimizationDirection = "Maximize",
            NumLeavesMin = 10,
            NumLeavesMax = 100
        )
    )

if __name__ == "__main__":
    main()