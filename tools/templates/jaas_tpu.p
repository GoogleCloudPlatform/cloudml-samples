# TODO(user): change this
args.model_dir = 'gs://your-gcs-bucket'

# Get hostname from environment using ipython magic.
# This returns a list.
hostname = !hostname

args.tpu = hostname[0]
args.use_tpu = True

# Use gcloud command line tool to create a TPU in the same zone as the VM instance.
! gcloud compute tpus create `hostname` \
  --zone `gcloud compute instances list --filter="name=$(hostname)" --format 'csv[no-heading](zone)'`\
  --network default \
  --range 10.101.1.0 \
  --version 1.13
