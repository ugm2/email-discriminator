import prefect


def call_flow(
    deployment_name: str = "train-flow/email_discriminator-train",
    parameters: dict = {"model_stage": "Staging"},
):
    """
    Calls a deployed Prefect flow.

    Args:
      deployment_name: The name of the deployment.
      parameters: The parameters to pass to the flow.

    Returns:
      The flow run of the flow execution.
    """

    # Run the deployed flow
    flow_run = prefect.deployments.run_deployment(
        name=deployment_name, parameters=parameters, timeout=0
    )

    return flow_run
