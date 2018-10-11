# This is only for when running on Colab:
# Get the dependency .py files, if any.
import sys
if 'google.colab' in sys.modules:
    ! git clone https://github.com/GoogleCloudPlatform/cloudml-samples.git
    ! cp cloudml-samples/{path}/* .
