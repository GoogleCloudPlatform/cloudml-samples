FROM gcr.io/r-on-gcp/caret_base

RUN mkdir -p /root
COPY model_trainer.R /root
WORKDIR /root

CMD ["Rscript", "model_trainer.R"]