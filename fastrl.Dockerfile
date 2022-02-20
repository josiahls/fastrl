#FROM pytorch/pytorch
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

ENV CONTAINER_USER fastrl_user
ENV CONTAINER_GROUP fastrl_group
ENV CONTAINER_UID 1000
# Add user to conda
RUN addgroup --gid $CONTAINER_UID $CONTAINER_GROUP && \
    adduser --uid $CONTAINER_UID --gid $CONTAINER_UID $CONTAINER_USER --disabled-password  && \
    mkdir -p /opt/conda && chown $CONTAINER_USER /opt/conda

RUN apt-get update && apt-get install -y software-properties-common rsync curl
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0 && apt-add-repository https://cli.github.com/packages


RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

RUN apt-get update && apt-get install -y git libglib2.0-dev graphviz libxext6 libsm6 libxrender1 python-opengl xvfb nano gh && apt-get update

RUN pip install albumentations \
    catalyst \
    captum \
    "fastprogress>=0.1.22" \
    graphviz \
    jupyter \
    kornia \
    matplotlib \
    "nbconvert<6"\
    nbdev \
    neptune-client \
    opencv-python \
    pandas \
    pillow \
    pyarrow \
    pydicom \
    pyyaml \
    scikit-learn \
    scikit-image \
    scipy \
    "sentencepiece<0.1.90" \
    spacy \
    tensorboard \
    wandb \
    jupyterlab \
    watchdog[watchmedo] \
    ipywidgets \
    moviepy \
    pygifsicle \
    aquirdturtle_collapsible_headings \
    xeus-python \
    matplotlib_inline

RUN pip install fastai --no-dependencies

WORKDIR /home/$CONTAINER_USER
RUN /bin/bash -c "echo 'break cache'"
ARG BUILD=dev
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && conda install -c conda-forge nodejs==15.14.0 line_profiler && jupyter labextension install jupyterlab-plotly && pip install plotly rich[jupyter]; fi"
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && git clone https://github.com/benelot/pybullet-gym.git && cd pybullet-gym && pip install -e .; fi"

RUN pip install --pre torch -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
RUN pip install -e git+https://github.com/pytorch/data#egg=torchdata

RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/bin
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/lib/python3.*/site-packages/torch/utils/data/datapipes
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /home/$CONTAINER_USER

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/themes.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/shortcuts.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/tracker.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/

USER $CONTAINER_USER
WORKDIR /home/$CONTAINER_USER
ENV PATH="/home/$CONTAINER_USER/.local/bin:${PATH}"

RUN git clone https://github.com/fastai/fastai.git --depth 1 \
        && git clone https://github.com/fastai/fastcore.git --depth 1 \
        && git clone https://github.com/josiahls/fastrl.git --depth 1
RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastai && pip install .  && cd ../fastcore && pip install .  && cd ../fastrl && pip install . ; fi"
# Note that we are not installing the .dev dependencies for fastai or fastcore
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastai && pip install -e \".[dev]\" --user  && cd ../fastcore && pip install -e \".[dev]\" --user  && cd ../fastrl && pip install -e \".[dev]\" --user ; fi"

RUN echo '#!/bin/bash\npip install -e .[dev]  && xvfb-run -s "-screen 0 1400x900x24" jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.password=''' >> run_jupyter.sh

USER $CONTAINER_USER
RUN /bin/bash -c "cd fastrl && pip install -e ."
RUN chmod u+x run_jupyter.sh