import argparse
import logging
import os

from prefect.deployments import Deployment
from prefect.filesystems import GCS
from prefect.server.schemas.schedules import CronSchedule
from prefect_gcp.cloud_run import CloudRunJob
from rich.logging import RichHandler

from email_discriminator.workflows.predict import predict_flow
from email_discriminator.workflows.train import train_flow

# Get the logger level from environment variables. Default to WARNING if not set.
LOGGER_LEVEL = os.getenv("LOGGER_LEVEL", "WARNING")
logging.basicConfig(level=LOGGER_LEVEL, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("Flow Deployment")


def deploy(
    flow,
    name,
    version,
    gcs_block,
    cloud_run_block,
    work_queue_name="main",
    work_pool_name="google-pool",
    schedule=None,
    flow_parameters={},
):
    logging.info(f"Deploying {name} version {version} to Cloud Run...")
    deployment = Deployment.build_from_flow(
        flow=flow,
        name=name,
        version=version,
        storage=gcs_block,
        work_queue_name=work_queue_name,
        work_pool_name=work_pool_name,
        infrastructure=cloud_run_block,
        schedule=schedule,
        parameters=flow_parameters,
    )
    deployment.apply()
    logging.info(f"Deployment of {name} version {version} completed successfully.")


def main(args):
    gcs_block = GCS.load("email-discriminator-bucket")
    cloud_run_block = CloudRunJob.load("email-discriminator-cloud-run")
    user_schedule = CronSchedule(cron=args.cron, timezone="Europe/Madrid")

    if args.workflow == "all" or args.workflow == "predict":
        deploy(
            predict_flow,
            "email_discriminator-predict",
            args.version,
            gcs_block,
            cloud_run_block,
            schedule=user_schedule,
            flow_parameters={"do_delete_emails": args.predict_delete_emails},
        )

    if args.workflow == "all" or args.workflow == "train":
        deploy(
            train_flow,
            "email_discriminator-train",
            args.version,
            gcs_block,
            cloud_run_block,
            schedule=None,  # Train flow does not have a schedule, it runs from the UI
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy workflows to Cloud Run.")
    parser.add_argument(
        "--workflow",
        choices=["predict", "train", "all"],
        default="all",
        nargs="?",
        help="The workflow to deploy. 'all' will deploy both workflows.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        nargs="?",
        help="The version number for the deployment.",
    )
    parser.add_argument(
        "--cron", default="0 9 * * 4", help="Cron string to specify the flow schedule."
    )
    parser.add_argument("--predict-delete_emails", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
