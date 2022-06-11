FROM python:3.8.3-slim as package

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN apt-get update && apt-get install -y \
curl && apt-get install -y unzip

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

WORKDIR ${HOME}
ENV PATH=${PATH}:.local/bin
RUN pip install --user --no-cache-dir poetry && \
    poetry config virtualenvs.in-project true
RUN poetry install
ENV PYTHONPATH "${PYTHONPATH}:${HOME}"
ENV PATH=${PATH}:.venv/bin

ENTRYPOINT []