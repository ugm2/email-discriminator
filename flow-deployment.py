import argparse

from prefect.deployments import Deployment
from prefect.filesystems import GCS
from prefect_gcp.cloud_run import CloudRunJob

from email_discriminator.workflows.predict import predict_flow
from email_discriminator.workflows.train import train_flow


def deploy(flow, name, version, gcs_block, cloud_run_block, work_queue_name="main"):
    deployment = Deployment.build_from_flow(
        flow=flow,
        name=name,
        version=version,
        storage=gcs_block,
        work_queue_name=work_queue_name,
        infrastructure=cloud_run_block,
    )
    deployment.apply()


def main(args):
    gcs_block = GCS.load("email-discriminator-bucket")
    cloud_run_block = CloudRunJob.load("email-discriminator-cloud-run")

    if args.workflow == "all" or args.workflow == "predict":
        deploy(
            predict_flow,
            "email_discriminator-predict",
            args.version,
            gcs_block,
            cloud_run_block,
        )

    if args.workflow == "all" or args.workflow == "train":
        deploy(
            train_flow,
            "email_discriminator-train",
            args.version,
            gcs_block,
            cloud_run_block,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy workflows to Cloud Run.")
    parser.add_argument(
        "workflow",
        choices=["predict", "train", "all"],
        default="all",
        nargs="?",
        help="The workflow to deploy. 'all' will deploy both workflows.",
    )
    parser.add_argument(
        "version",
        type=int,
        default=1,
        nargs="?",
        help="The version number for the deployment.",
    )
    args = parser.parse_args()
    main(args)
