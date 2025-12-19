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
# import sys
# import traceback
# from pipelines._utils import get_pipeline_driver, convert_struct, get_pipeline_custom_tags

"""Example workflow pipeline script for abalone pipeline.

                                               . -ModelStep
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os
import os
import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.conditions import ConditionGreaterThan,ConditionEquals
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.tuner import (
    IntegerParameter,
    CategoricalParameter,
    ContinuousParameter,
    HyperparameterTuner,
)
from sagemaker.workflow.steps import TuningStep
from sagemaker.estimator import Estimator
import random
from sagemaker import image_uris, model_uris, script_uris
from sagemaker.workflow.steps import TrainingStep
import json

from sagemaker import hyperparameters
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean
)
from sagemaker.workflow.functions import Join


print(os.getcwd())


# BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="AbalonePackageGroup",
    pipeline_name="AbalonePipeline",
    base_job_prefix="Abalone",
    processing_instance_type="ml.m5.xlarge",
    training_instance_type="ml.m5.xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on abalone data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    import json
    # Read Default Config from Configuration File
    with open("train_config.json", "r") as f:
        training_config=json.load(f)
    
    # LightGBM tunable parameters for SageMaker Pipelines
    learning_rate_min = ParameterFloat(name="LearningRateMin", default_value=float(training_config["LearningRateMin"]))
    learning_rate_max = ParameterFloat(name="LearningRateMax", default_value=float(training_config["LearningRateMax"]))

    num_boost_round_min = ParameterInteger(name="NumberOfBoostRoundMin", default_value=int(training_config["NumberOfBoostRoundMin"]))
    num_boost_round_max = ParameterInteger(name="NumberOfBoostRoundMax", default_value=int(training_config["NumberOfBoostRoundMax"]))

    num_leaves_min = ParameterInteger(name="NumLeavesMin", default_value=int(training_config["NumLeavesMin"]))
    num_leaves_max = ParameterInteger(name="NumLeavesMax", default_value=int(training_config["NumLeavesMax"]))

    feature_fraction_min = ParameterFloat(name="FeatureFractionMin", default_value=float(training_config["FeatureFractionMin"]))
    feature_fraction_max = ParameterFloat(name="FeatureFractionMax", default_value=float(training_config["FeatureFractionMax"]))

    bagging_fraction_min = ParameterFloat(name="BaggingFractionMin", default_value=float(training_config["BaggingFractionMin"]))
    bagging_fraction_max = ParameterFloat(name="BaggingFractionMax", default_value=float(training_config["BaggingFractionMax"]))

    bagging_freq_min = ParameterInteger(name="BaggingFreqMin", default_value=int(training_config["BaggingFreqMin"]))
    bagging_freq_max = ParameterInteger(name="BaggingFreqMax", default_value=int(training_config["BaggingFreqMax"]))

    max_depth_min = ParameterInteger(name="MaxDepthMin", default_value=int(training_config["MaxDepthMin"]))
    max_depth_max = ParameterInteger(name="MaxDepthMax", default_value=int(training_config["MaxDepthMax"]))

    min_data_in_leaf_min = ParameterInteger(name="MinDataInLeafMin", default_value=int(training_config["MinDataInLeafMin"]))
    min_data_in_leaf_max = ParameterInteger(name="MinDataInLeafMax", default_value=int(training_config["MinDataInLeafMax"]))

    tuner_objective_metric = ParameterString(name="TunerObjectiveMetric", default_value=training_config["TunerObjectiveMetric"])
    tuner_metric_definition = ParameterString(name="TunerMetricDefinition", default_value=training_config["TunerMetricDefinition"])
    algo_metric = ParameterString(name="AlgorithmMetric", default_value=training_config["AlgorithmMetric"])

    max_tuning_jobs = ParameterInteger(name="MaxTuningJobs", default_value=int(training_config["MaxTuningJobs"]))
    max_tuning_parallel_job = ParameterInteger(name="TuningParallelJobs", default_value=int(training_config["TuningParallelJobs"]))
    tuning_strategy = ParameterString(name="TuningStrategy", default_value=training_config["TuningStrategy"], enum_values=["Bayesian", "Random", "Grid", "Hyperband"])
    optimization_direction = ParameterString(name="OptimizationDirection", default_value=training_config["OptimizationDirection"], enum_values=["Maximize", "Minimize"])
    supervised_training_task = ParameterString(name="TrainingTask", default_value=training_config["TrainingTask"], enum_values=["classification", "regression"])

    # Infra Parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=int(training_config["ProcessingInstanceCount"]))
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value=training_config["ProcessingInstanceType"])
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value=training_config["TrainingInstanceType"])
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=int(training_config["TrainingInstanceCount"]))
    training_volume_size = ParameterInteger(name="TrainingVolumeSize", default_value=int(training_config["TrainingVolumeSize"]))
    processing_volume_size = ParameterInteger(name="ProcessingVolumeSize", default_value=int(training_config["ProcessingVolumeSize"]))

    # Artifacts location Parameters
    model_approval_status = ParameterString(name="ModelApprovalStatus", default_value=training_config["ModelApprovalStatus"])
    model_output_bucket = ParameterString(name="ModelOutput", default_value=training_config["ModelOutput"])
    train_output_bucket = ParameterString(name="TrainOutput", default_value=training_config["TrainOutput"])
    validation_output_bucket = ParameterString(name="ValidationOutput", default_value=training_config["ValidationOutput"])
    test_output_bucket = ParameterString(name="TestOutput", default_value=training_config["TestOutput"])
    s3_input_data_location = ParameterString(name="S3InputDataURI", default_value=training_config["S3InputDataURI"])

    # Mlflow
    ml_flow_arn = ParameterString(name="MLflow", default_value=training_config["MLflow"])

    model_evaluation_threshold = ParameterFloat(name="EvalThreshold", default_value=float(training_config["EvalThreshold"]))
    data_split_ratio = ParameterString(name="DataSplitRatio", default_value=training_config["DataSplitRatio"])

    
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)

   
    # processing step for feature engineering
    framework_version = "1.0-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=framework_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        volume_size_in_gb = processing_volume_size,
        base_job_name="sklearn-pre-process",
        role=role,
        sagemaker_session=pipeline_session,
    )
    processor_args = sklearn_processor.run(
        inputs=[
          ProcessingInput(source=s3_input_data_location, destination="/opt/ml/processing/input"),  
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train",\
                             destination = train_output_bucket),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation",\
                            destination = validation_output_bucket),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test",\
                            destination = test_output_bucket)
        ],
        code="pipeline_scripts/churn_preprocess.py",
        arguments =[
            "--split-ratio",data_split_ratio
        ],
    )
    step_process = ProcessingStep(name="LightGBMDataPreProcess", step_args=processor_args)

    # training step for generating model artifacts
    
    
    train_model_id, train_model_version, train_scope = "lightgbm-classification-model", "*", "training"


    # # Retrieve the docker image
    train_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        model_id=train_model_id,
        model_version=train_model_version,
        image_scope=train_scope,
        instance_type=training_instance_type,
    )

    # Retrieve the pre-trained model tarball to further fine-tune
    train_model_uri = model_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
    )
    # Retrieve the default hyper-parameters for fine-tuning the model
    hyperparameters = hyperparameters.retrieve_default(
        model_id=train_model_id, model_version=train_model_version
    )

    # [Optional] Override default hyperparameters with custom values
    hyperparameters["num_boost_round"] = "200"
    hyperparameters["metric"] = algo_metric # pipeline parameter

    # Recommended for distributed training
    hyperparameters["tree_learner"] = "voting" 
    del hyperparameters["early_stopping_rounds"]

    print(hyperparameters)

    # Create SageMaker Estimator instance
    tabular_estimator = Estimator(
        role=role,
        image_uri=train_image_uri,
        source_dir= "model_cat", 
        model_uri=train_model_uri,
        entry_point="train.py", 
        instance_count= training_instance_count,  # pipeline paramter
        volume_size=training_volume_size,  # pipeline paramter
        instance_type=training_instance_type, # pipeline paramter
        max_run=360000,
        hyperparameters=hyperparameters,
        output_path=model_output_bucket,
        sagemaker_session=pipeline_session, # Tells it its part of a Sagemaker Pipeline and not to execute individually
        environment={"MLFLOW_TRACKING_ARN": ml_flow_arn}, # pipeline paramter
        keep_alive_period_in_seconds = 1000 #Keep instance warm for fast experimentation iteration else experience cold start for each trials (note you will incur cost of warm instances)
    )
    
    
    from sagemaker.tuner import ContinuousParameter, IntegerParameter, HyperparameterTuner

    # Define hyperparameter ranges (Pipeline parameters)
    hyperparameter_ranges_lgb = {
        "learning_rate": ContinuousParameter(learning_rate_min , learning_rate_max , scaling_type="Auto"),
        "num_boost_round": IntegerParameter(num_boost_round_min , num_boost_round_max),
        "num_leaves": IntegerParameter(num_leaves_min , num_leaves_max),
        "feature_fraction": ContinuousParameter(feature_fraction_min, feature_fraction_max),
        "bagging_fraction": ContinuousParameter(bagging_fraction_min, bagging_fraction_max),
        "bagging_freq": IntegerParameter(bagging_freq_min, bagging_freq_max),
        "max_depth": IntegerParameter(max_depth_min, max_depth_max),
        "min_data_in_leaf": IntegerParameter(min_data_in_leaf_min, min_data_in_leaf_max),
    }



    tuner = HyperparameterTuner(
        estimator = tabular_estimator,
        objective_metric_name = tuner_objective_metric, # pipeline paramter
        hyperparameter_ranges = hyperparameter_ranges_lgb,  # pipeline paramter
        metric_definitions = [{"Name": tuner_objective_metric, "Regex": Join(on=':',values=[tuner_objective_metric ," ([0-9\\.]+)" ] )}], # pipeline paramter
        max_jobs=max_tuning_jobs, # pipeline paramter
        max_parallel_jobs=max_tuning_parallel_job, # pipeline paramter
        objective_type=optimization_direction, # pipeline paramter
        strategy = tuning_strategy # pipeline paramter
    ) 

    # Here we create an implicit dependencies between the processing step and Tuning step
    hpo_args = tuner.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_tuning = TuningStep(
        name="LightGBMClassifierHyperParameterTuning",
        step_args=hpo_args,
    )
    
    
    
    
    
    train_model_id, train_model_version, train_scope = "lightgbm-regression-model", "*", "training"


    # Retrieve the docker image
    train_image_uri = image_uris.retrieve(
        region=None,
        framework=None,
        model_id=train_model_id,
        model_version=train_model_version,
        image_scope=train_scope,
        instance_type=training_instance_type,
    )

    # Retrieve the pre-trained model tarball to further fine-tune
    train_model_uri = model_uris.retrieve(
        model_id=train_model_id, model_version=train_model_version, model_scope=train_scope
    )
    # Retrieve the default hyper-parameters for fine-tuning the model
    hyperparameters = hyperparameters.retrieve_default(
        model_id=train_model_id, model_version=train_model_version
    )

    # [Optional] Override default hyperparameters with custom values
    hyperparameters["num_boost_round"] = "200"
    hyperparameters["metric"] = algo_metric

    # Recommended for distributed training
    hyperparameters["tree_learner"] = "voting" 
    del hyperparameters["early_stopping_rounds"]
    print(hyperparameters)

    # Create SageMaker Estimator instance
    reg_estimator = Estimator(
        role=role,
        image_uri=train_image_uri,
        source_dir= "model_reg", 
        model_uri=train_model_uri,
        entry_point="train.py", 
        instance_count= training_instance_count,  
        volume_size=training_volume_size, 
        instance_type=training_instance_type,
        max_run=360000,
        hyperparameters=hyperparameters,
        output_path=model_output_bucket,
        sagemaker_session=pipeline_session, # Tells it its part of a Sagemaker Pipeline and not to execute individually
        environment={"MLFLOW_TRACKING_ARN": ml_flow_arn},
        keep_alive_period_in_seconds = 1000 #Keep instance warm for fast experimentation iteration else experience cold start for each trials (note you will incur cost of warm instances)
    )
    
   
    # Define hyperparameter ranges
    hyperparameter_ranges_lgb = {
        "learning_rate": ContinuousParameter(learning_rate_min , learning_rate_max , scaling_type="Auto"),
      "num_boost_round": IntegerParameter(num_boost_round_min , num_boost_round_max),
        "num_leaves": IntegerParameter(num_leaves_min , num_leaves_max),
        "feature_fraction": ContinuousParameter(feature_fraction_min, feature_fraction_max),
        "bagging_fraction": ContinuousParameter(bagging_fraction_min, bagging_fraction_max),
        "bagging_freq": IntegerParameter(bagging_freq_min, bagging_freq_max),
        "max_depth": IntegerParameter(max_depth_min, max_depth_max),
        "min_data_in_leaf": IntegerParameter(min_data_in_leaf_min, min_data_in_leaf_max),
    }

    tuner_reg = HyperparameterTuner(
        estimator = reg_estimator,
        objective_metric_name = tuner_objective_metric,
        hyperparameter_ranges = hyperparameter_ranges_lgb, 
        metric_definitions = [{"Name": tuner_objective_metric, "Regex": Join(on=':',values=[tuner_objective_metric ," ([0-9\\.]+)" ] )}],
        max_jobs=max_tuning_jobs,
        max_parallel_jobs=max_tuning_parallel_job, 
        objective_type=optimization_direction,
        strategy = tuning_strategy)

    # Here we create an implicit dependencies between the processing step and Tuning step
    hpo_args_reg = tuner_reg.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )

    step_tuning_reg = TuningStep(
        name="LightGBMHyperParameterTuningRegression",
        step_args=hpo_args_reg,
    )



    # Define Condition. Here we evaluate the condition based on the training task passed as a pipeline parameter 
    cond_task = ConditionEquals(
        left=supervised_training_task,
        right="classification",
    )

    # Condition Step
    """
    Here we create a condition syep to swith the branch based on training task type. 
    Run Classifier tuner if its a classification model or Regression tuner if its a regression model
    """
    step_cond = ConditionStep(
        depends_on = [step_process], # Depends on the processing step
        name="TrainingTaskTypes",
        conditions=[cond_task], 
        if_steps=[step_tuning], # If condition is true
        else_steps=[step_tuning_reg] # If condition is false
    )    

    
    
    
    import json
    from sagemaker.workflow.pipeline import Pipeline

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            # LightGBM tunable parameters
            learning_rate_min,
            learning_rate_max,
            num_leaves_min,
            num_leaves_max,
            feature_fraction_min,
            feature_fraction_max,
            bagging_fraction_min,
            bagging_fraction_max,
            bagging_freq_min,
            bagging_freq_max,
            max_depth_min,
            max_depth_max,
            min_data_in_leaf_min,
            min_data_in_leaf_max,
            num_boost_round_max,
            num_boost_round_min,

            # Other parameters
            processing_volume_size,
            training_volume_size,
            tuner_metric_definition,
            tuner_objective_metric,
            algo_metric,
            processing_instance_count,
            processing_instance_type,
            training_instance_type,
            training_instance_count,
            model_approval_status,
            model_output_bucket,
            train_output_bucket,
            validation_output_bucket,
            test_output_bucket,
            max_tuning_jobs,
            max_tuning_parallel_job,
            tuning_strategy,
            optimization_direction,
            ml_flow_arn,
            supervised_training_task,
            model_evaluation_threshold,
            s3_input_data_location,
            data_split_ratio,
        ],
        steps=[step_cond], # we pass only the condition step as we have declared all steps as dependencies to the condition step
    )

    # definition = json.loads(pipeline.definition())
    # print(definition)
    
    return pipeline

def main():  # pragma: no cover
    pipeline = get_pipeline()
    pipeline.upsert(role_arn="arn:aws:iam::734584155256:role/service-role/AmazonSageMaker-ExecutionRole-20221023T222844")
    # start Pipeline execution
    pipeline.start()


if __name__ == "__main__":
    main()

# def main():  # pragma: no cover
<<<<<<< HEAD
#     import json
#     # from sagemaker.workflow.pipeline import Pipeline
#     # with open("train_config.json", "r") as f:
#     #     training_config = json.load(f)
#     # pipeline = Pipeline(
#     #     name=training_config["PipelineName"])
#     # execution = pipeline.start()
#     pipeline = pipeline_files.get_pipeline()
#     pipeline.upsert(role_arn="arn:aws:iam::734584155256:role/service-role/AmazonSageMaker-ExecutionRole-20221023T222844")
#     # print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")
# # #     """The main harness that creates or updates and runs the pipeline.


# # import sys
# # import argparse
# # from abalone.pipeline import get_pipeline  # Replace 'your_module' with the actual module name

# # def main():
# #     parser = argparse.ArgumentParser(description="Run the Abalone Pipeline")
# #     parser.add_argument("--region", type=str, required=True, help="AWS region")
# #     parser.add_argument("--role", type=str, help="IAM role ARN")
# #     parser.add_argument("--default-bucket", type=str, help="Default S3 bucket")
# #     parser.add_argument("--pipeline-name", type=str, default="AbalonePipeline", help="Pipeline name")
# #     parser.add_argument("--base-job-prefix", type=str, default="Abalone", help="Base job prefix")
# #     args = parser.parse_args()

# #     pipeline = get_pipeline(
# #         region=args.region,
# #         role=args.role,
# #         default_bucket=args.default_bucket,
# #         pipeline_name=args.pipeline_name,
# #         base_job_prefix=args.base_job_prefix,
# #     )

# #     # You can add additional logic here to work with the pipeline object
# #     print(f"Pipeline created with name: {pipeline.name}")

# if __name__ == "__main__":
#     main()

=======
#     pipeline = get_pipeline()
#     pipeline.upsert(role_arn="arn:aws:iam::734584155256:role/service-role/AmazonSageMaker-ExecutionRole-20221023T222844")
#     # start Pipeline execution
#     pipeline.start()


# if __name__ == "__main__":
#     main()

# # def main():  # pragma: no cover
# #     import json
# #     # from sagemaker.workflow.pipeline import Pipeline
# #     # with open("train_config.json", "r") as f:
# #     #     training_config = json.load(f)
# #     # pipeline = Pipeline(
# #     #     name=training_config["PipelineName"])
# #     # execution = pipeline.start()
# #     pipeline = pipeline_files.get_pipeline()
# #     pipeline.upsert(role_arn="arn:aws:iam::734584155256:role/service-role/AmazonSageMaker-ExecutionRole-20221023T222844")
# #     # print(f"\n###### Execution started with PipelineExecutionArn: {execution.arn}")
# # # #     """The main harness that creates or updates and runs the pipeline.


# # # import sys
# # # import argparse
# # # from abalone.pipeline import get_pipeline  # Replace 'your_module' with the actual module name

# # # def main():
# # #     parser = argparse.ArgumentParser(description="Run the Abalone Pipeline")
# # #     parser.add_argument("--region", type=str, required=True, help="AWS region")
# # #     parser.add_argument("--role", type=str, help="IAM role ARN")
# # #     parser.add_argument("--default-bucket", type=str, help="Default S3 bucket")
# # #     parser.add_argument("--pipeline-name", type=str, default="AbalonePipeline", help="Pipeline name")
# # #     parser.add_argument("--base-job-prefix", type=str, default="Abalone", help="Base job prefix")
# # #     args = parser.parse_args()

# # #     pipeline = get_pipeline(
# # #         region=args.region,
# # #         role=args.role,
# # #         default_bucket=args.default_bucket,
# # #         pipeline_name=args.pipeline_name,
# # #         base_job_prefix=args.base_job_prefix,
# # #     )

# # #     # You can add additional logic here to work with the pipeline object
# # #     print(f"Pipeline created with name: {pipeline.name}")

# # if __name__ == "__main__":
# #     main()

>>>>>>> 86ddb760d9946608fef9aaf24adbfede21856b40
    
