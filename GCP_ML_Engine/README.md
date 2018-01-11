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

# Useful links
https://cloud.google.com/ml-engine/docs/packaging-trainer
https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training