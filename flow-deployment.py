from prefect.deployments import Deployment
from prefect.filesystems import GCS
from prefect_gcp.cloud_run import CloudRunJob

from email_discriminator.workflows.predict import predict_flow

gcs_block = GCS.load("email-discriminator-bucket")
cloud_run_block = CloudRunJob.load("email-discriminator-cloud-run")

deployment = Deployment.build_from_flow(
    flow=predict_flow,
    name="email_discriminator-predict",
    version=1,
    storage=gcs_block,
    work_queue_name="main",
    infrastructure=cloud_run_block,
)

deployment.apply()
