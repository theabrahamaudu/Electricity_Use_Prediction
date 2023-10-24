from src.data.preprocess import preprocessPipeline
from src.utils.pipeline_log_config import pipeline as logger

if __name__ == "__main__":
    STAGE_01 = "Preprocess"
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

    STAGE_02 = "Train"

