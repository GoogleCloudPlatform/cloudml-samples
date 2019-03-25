# Use gcloud command line tool to delete the TPU.
! gcloud compute tpus delete `hostname` \
  --zone `gcloud compute instances list --filter="name=$(hostname)" --format 'csv[no-heading](zone)'`\
  --quiet
