FROM gcr.io/r-on-gcp/caret_base

RUN mkdir -p /root
COPY model_prediction.R /root
COPY model_api.R /root
WORKDIR /root

ENV PORT 8080

EXPOSE 8080
                         
ENTRYPOINT ["Rscript", "model_api.R"]
