# Base Image   
FROM continuumio/miniconda3:latest

RUN conda install -c conda-forge nb_conda_kernels
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    ninja-build \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /app

COPY environment.yml .
COPY ./stateAvoid ./stateAvoid_src

RUN conda env create -f environment.yml

ENV PATH /opt/conda/envs/sa_env/bin:$PATH

RUN /opt/conda/envs/sa_env/bin/pip install git+https://github.com/hklarner/pyboolnet

RUN echo "source activate sa_env" >> ~/.bashrc

WORKDIR /app/stateAvoid

CMD ["python", "execute.py"]