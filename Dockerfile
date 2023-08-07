FROM prefecthq/prefect:2.11.0-python3.10

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in pyproject.toml
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-interaction --no-ansi

# Add the poetry scripts to the PATH
ENV PATH="${PATH}:/root/.local/bin"
