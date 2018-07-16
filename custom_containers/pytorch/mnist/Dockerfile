# Dockerfile
FROM python:2.7.15-jessie
WORKDIR /root

# Installs pytorch and torchvision.
RUN pip install torch torchvision

# This is for cloud ml hyperparameter tuning.
RUN pip install cloudml-hypertune

# Installs google cloud sdk, this is mostly for using GCS to store the model.
RUN wget -nv https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup
# Path configuration
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

RUN mkdir /root/trainer
COPY trainer/mnist.py /root/trainer/mnist.py

ENTRYPOINT ["python", "trainer/mnist.py"]
