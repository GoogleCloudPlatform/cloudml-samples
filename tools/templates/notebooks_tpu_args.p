# TODO(user): change this
args.model_dir = 'gs://your-gcs-bucket'

# Get hostname from environment using ipython magic.
# This returns a list.
hostname = !hostname

args.tpu = hostname[0]
args.use_tpu = True
