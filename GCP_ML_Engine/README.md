## MEANINGFUL
Same project, but fitting the ML Engine requirements

# Install
If using the gcloud command line tool, setup will run automatically.
To run the setup.py file manually : python setup.py sdist

# Usage
- Use the gcloud tool to package and upload:
gcloud ml-engine jobs submit training JOB (The name for your job, which must be unique within your project)
    --module-name=MODULE_NAME (name of your application's main module using your package's namespace dot notation)
    [--config=CONFIG]
    [--job-dir=JOB_DIR]
    [--labels=[KEY=VALUE,…]]
    [--package-path=PACKAGE_PATH] (path to the root directory of your trainer application)
    [--packages=[PACKAGE,…]]
    [--region=REGION]
    [--runtime-version=RUNTIME_VERSION]
    [--scale-tier=SCALE_TIER]
    [--staging-bucket=STAGING_BUCKET] (Cloud Storage location that you want the tool to use to stage your training and dependency packages)
    [--async     | --stream-logs]
    [GCLOUD_WIDE_FLAG …] [-- USER_ARGS …]
    
    
- How to upload and launch on GCP:
    - Take the project packaged as specified in the tutorial (tar+ gz)
    - Upload it on GCP console
    - Specify variables
        - now=$(date +"%Y%m%d_%H%M%S")  then  JOB_NAME="census_$now"
        - $MAIN_TRAINER_MODULE= trainer.task
    - 
    
 gcloud ml-engine jobs submit training $JOB_NAME --staging-bucket "gs://meaningful_output" --module-name $MAIN_TRAINER_MODULE --region europe-west1 --packages "GCP_M
L_Engine.tar.gz"

# Useful links
https://cloud.google.com/ml-engine/docs/packaging-trainer
https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training
https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3