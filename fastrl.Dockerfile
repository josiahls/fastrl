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

RUN apt-get update && apt-get install -y git libglib2.0-dev graphviz libxext6 libsm6 libxrender1 python-opengl xvfb nano gh tree && apt-get update

WORKDIR /home/$CONTAINER_USER
# Install Primary Pip Reqs
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/pip_requirements.txt /home/$CONTAINER_USER/extra/pip_requirements.txt
RUN pip install -r extra/pip_requirements.txt

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/requirements.txt /home/$CONTAINER_USER/extra/requirements.txt
RUN echo "break cache" 
# RUN pip install fastai>=2.7.10 --no-dependencies
RUN pip install -r extra/requirements.txt && \
       pip uninstall -y torch && \
           pip install --pre torch torchdata --extra-index-url https://download.pytorch.org/whl/nightly/cu113 --upgrade
RUN pip show torch torchdata

# Install Dev Reqs
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/dev_requirements.txt /home/$CONTAINER_USER/extra/dev_requirements.txt
ARG BUILD=dev
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && pip install -r extra/dev_requirements.txt ; fi"
# RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && conda install -c conda-forge nodejs==15.14.0 line_profiler && jupyter labextension install jupyterlab-plotly; fi"

RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/bin
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/lib/python3.*/site-packages/torch/utils/data/datapipes
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /home/$CONTAINER_USER

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/themes.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/shortcuts.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/shortcuts-extension/
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/tracker.jupyterlab-settings /home/$CONTAINER_USER/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/

RUN apt-get install sudo
RUN nbdev_install_quarto
RUN pip install typing-extensions==4.1.1

USER $CONTAINER_USER
WORKDIR /home/$CONTAINER_USER
ENV PATH="/home/$CONTAINER_USER/.local/bin:${PATH}"

RUN git clone https://github.com/josiahls/fastrl.git --depth 1
RUN git clone https://github.com/fastai/fastai.git --depth 1 && cd fastai && pip install . --no-dependencies
RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastrl && pip install . --no-dependencies; fi"
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastrl && pip install -e \".[dev]\" --user --no-dependencies ; fi"

RUN echo '#!/bin/bash\npip install -e .[dev] --no-dependencies && xvfb-run -s "-screen 0 1400x900x24" jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.password=''' >> run_jupyter.sh

USER $CONTAINER_USER
RUN /bin/bash -c "cd fastrl && pip install -e . --no-dependencies"
RUN chmod u+x run_jupyter.sh
