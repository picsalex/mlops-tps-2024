from mlflow.server.auth.client import AuthServiceClient

# The script needs the following environment variables:
# export MLFLOW_TRACKING_USERNAME=username
# export MLFLOW_TRACKING_PASSWORD=password

# client = AuthServiceClient("http://localhost:5001")

# Create a new user
# client.create_user("user", "password")

# Set a user as Admin
# client.update_user_admin(username="user", is_admin=True)

# Delete a user
# client.delete_user("admin")
# client.update_user_password("user", "password")

# mlflow.log_input(dataset=Dataset(source=HTTPDatasetSource()))
