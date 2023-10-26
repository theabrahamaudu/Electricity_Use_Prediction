"""
Pipeline Module

This module defines a pipeline with multiple stages:
1. Preprocess Data
2. Train Model
3. Evaluate Model

Each stage can be controlled by specifying whether to execute it (True) or skip it (False).

Example:
    To run the pipeline, set the execution flags for the desired stages and run this module.

        STAGE_01 = ["Preprocess Data", True]
        STAGE_02 = ["Train Model", True]
        STAGE_03 = ["Evaluate Model", True]

    Setting a stage to True will execute it, while False will skip it.

Attributes:
    STAGE_01 (list): A list representing the first stage with a description and execution flag.
    STAGE_02 (list): A list representing the second stage with a description and execution flag.
    STAGE_03 (list): A list representing the third stage with a description and execution flag.
"""

from src.data.preprocess import preprocessPipeline
from src.models.train_model import modelPipeline
from src.utils.pipeline_log_config import pipeline as logger

if __name__ == "__main__":
    STAGE_01 = ["Preprocess Data", True]
    STAGE_02 = ["Train Model", False]
    STAGE_03 = ["Evaluate Model", True]
    logger.info(f"Running pipeline in stages: {STAGE_01}, {STAGE_02}, {STAGE_03}")

    if STAGE_01[1]:
        STAGE_01 = STAGE_01[0]
        logger.info(f">>>>>>>>>>>>> Starting Stage: {STAGE_01} <<<<<<<<<<<<<<")
        try:
            preprocessing = preprocessPipeline(
                raw_path="./data/raw",
                interim_path="./data/interim",
                processed_path="./data/processed"
            )
            preprocessing.trainPreprocess(
                skip_transformed=True,
                skip_grouped=True,
                skip_concatenated=True
            )
            logger.info(f">>>>>>>>>>>>> Finished Stage: {STAGE_01} <<<<<<<<<<<<<<")
        except Exception as e:
            logger.error(f"Failed to complete Stage: {STAGE_01}: {e}")
            raise e
    else:
        logger.info(f"Skipping Stage: {STAGE_01[0]}")

    if STAGE_02[1]:
        STAGE_02 = STAGE_02[0]
        logger.info(f">>>>>>>>>>>>> Starting Stage: {STAGE_02} <<<<<<<<<<<<<<")
        try:
            modeling = modelPipeline(
                processed_path="./data/processed",
                reports_path="./reports",
                models_path="./models",
                version= "0.2.4-fulldata"
            )
            model = modeling.build_model()
            modeling.save_model_diagram(model)
            training_output = modeling.train_model(model)
            modeling.save_model(training_output)
            logger.info(f">>>>>>>>>>>>> Finished Stage: {STAGE_02} <<<<<<<<<<<<<<")
        except Exception as e:
            logger.error(f"Failed to complete Stage: {STAGE_02}: {e}")
            raise e
    else:
        logger.info(f"Skipping Stage: {STAGE_02[0]}")
    
    if STAGE_03[1]:
        STAGE_03 = STAGE_03[0]
        logger.info(f">>>>>>>>>>>>> Starting Stage: {STAGE_03} <<<<<<<<<<<<<<")
        try:
            modeling = modelPipeline(
                processed_path="./data/processed",
                reports_path="./reports",
                models_path="./models",
                version= "pipeline-example"
            ) 
            model = modeling.load_model()
            yhat = modeling.evaluate_model(model)
            modeling.save_testplot(yhat)
            logger.info(f">>>>>>>>>>>>> Finished Stage: {STAGE_03} <<<<<<<<<<<<<<")
        except Exception as e:
            logger.error(f"Failed to complete Stage: {STAGE_03}: {e}")
            raise e
    else:
        logger.info(f"Skipping Stage: {STAGE_03[0]}")    
    

