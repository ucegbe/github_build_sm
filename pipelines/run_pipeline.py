# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""A CLI to create or update and run pipelines."""
# from __future__ import absolute_import

# import argparse
import json
from sagemaker.workflow.pipeline import Pipeline
# import sys
# import traceback
# from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags


def main():  # pragma: no cover
    
    with open("train_config.json", "r") as f:
        training_config = json.load(f)
    pipeline = Pipeline(
        name=training_config["PipelineName"])
    
    pipeline.start(
        parameters=dict(
            AlgorithmMetric="binary_error",
            TunerObjectiveMetric = "binary_error",
            TunerMetricDefinition="binary_error: ([0-9\\.]+)",
            OptimizationDirection = "Minimize"
        )
    )
    print(f"\n###### Execution started with PipelineExecutionArn: {pipeline.arn}")
#     """The main harness that creates or updates and runs the pipeline.


# import sys
# import argparse
# from abalone.pipeline import get_pipeline  # Replace 'your_module' with the actual module name

# def main():
#     parser = argparse.ArgumentParser(description="Run the Abalone Pipeline")
#     parser.add_argument("--region", type=str, required=True, help="AWS region")
#     parser.add_argument("--role", type=str, help="IAM role ARN")
#     parser.add_argument("--default-bucket", type=str, help="Default S3 bucket")
#     parser.add_argument("--pipeline-name", type=str, default="AbalonePipeline", help="Pipeline name")
#     parser.add_argument("--base-job-prefix", type=str, default="Abalone", help="Base job prefix")
#     args = parser.parse_args()

#     pipeline = get_pipeline(
#         region=args.region,
#         role=args.role,
#         default_bucket=args.default_bucket,
#         pipeline_name=args.pipeline_name,
#         base_job_prefix=args.base_job_prefix,
#     )

#     # You can add additional logic here to work with the pipeline object
#     print(f"Pipeline created with name: {pipeline.name}")

if __name__ == "__main__":
    main()

    
