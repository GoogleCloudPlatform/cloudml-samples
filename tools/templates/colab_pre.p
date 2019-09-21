# Only for when running on Colab:
import sys
if 'google.colab' in sys.modules:
    # Get the dependency .py files, if any.
    ! git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
    ! cp cloudml-samples/{path}/* .

    # Authenticate the user for better GCS access.
    # Copy verification code into the text field to continue.
    from google.colab import auth
    auth.authenticate_user()
