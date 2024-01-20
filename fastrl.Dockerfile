# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu18.04
# RUN  conda install python=3.8

ENV CONTAINER_USER fastrl_user
ENV CONTAINER_GROUP fastrl_group
ENV CONTAINER_UID 1000
# Add user to conda
RUN addgroup --gid $CONTAINER_UID $CONTAINER_GROUP && \
    adduser --uid $CONTAINER_UID --gid $CONTAINER_UID $CONTAINER_USER --disabled-password 
    #  && \
    # mkdir -p /opt/conda && chown $CONTAINER_USER /opt/conda

RUN apt-get update && apt-get install -y software-properties-common rsync curl
#RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0 && apt-add-repository https://cli.github.com/packages

RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null

RUN apt-get install -y python3.8-dev python3.8-distutils
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2 && update-alternatives --config python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.8 get-pip.py

RUN apt-get update && apt-get install -y git libglib2.0-dev graphviz libxext6 \
        libsm6 libxrender1 python-opengl xvfb nano gh tree wget libosmesa6-dev \
        libgl1-mesa-glx libglfw3 && apt-get update
 

WORKDIR /home/$CONTAINER_USER
# Install Primary Pip Reqs
# ENV PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/nightly/cu117
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/requirements.txt /home/$CONTAINER_USER/extra/requirements.txt
# Since we are using a custom fork of torchdata, we install torchdata as part of a submodule.
# RUN pip3 install -r extra/requirements.txt --pre --upgrade
RUN pip3 install torch>=2.0.0 
# --pre --upgrade
RUN pip3 show torch

COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/pip_requirements.txt /home/$CONTAINER_USER/extra/pip_requirements.txt
RUN pip3 install -r extra/pip_requirements.txt

WORKDIR /home/$CONTAINER_USER/fastrl
RUN git clone https://github.com/josiahls/data.git \
    && cd data && pip3 install -e .
WORKDIR /home/$CONTAINER_USER

# Install Dev Reqs
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP extra/dev_requirements.txt /home/$CONTAINER_USER/extra/dev_requirements.txt
ARG BUILD=dev
# Needed for gymnasium[all] when installing Box2d
RUN apt-get install swig3.0 && ln -s /usr/bin/swig3.0 /usr/bin/swig
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && pip3 install -r extra/dev_requirements.txt ; fi"
# RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz; fi"
# RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && conda install -c conda-forge nodejs==15.14.0 line_profiler && jupyter labextension install jupyterlab-plotly; fi"

# RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /opt/conda/bin
RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /usr/local/lib/python3.8/dist-packages/torch/utils/data/datapipes
# RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /usr/local/lib/python3.8/dist-packagess/mujoco_py
# RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /usr/local/lib/python3.8/dist-packages

RUN pip3 show torch

RUN chown $CONTAINER_USER:$CONTAINER_GROUP -R /home/$CONTAINER_USER

RUN apt-get install sudo
# Give user password-less sudo access
RUN echo "$CONTAINER_USER ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$CONTAINER_USER && \
    chmod 0440 /etc/sudoers.d/$CONTAINER_USER

RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then nbdev_install_quarto ; fi"
    
# RUN mkdir -p /home/$CONTAINER_USER/.mujoco \
#     && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
#     && tar -xf mujoco.tar.gz -C /home/$CONTAINER_USER/.mujoco \
#     && rm mujoco.tar.gz

# ENV LD_LIBRARY_PATH /home/$CONTAINER_USER/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so

USER $CONTAINER_USER
WORKDIR /home/$CONTAINER_USER
ENV PATH="/home/$CONTAINER_USER/.local/bin:${PATH}"

# RUN git clone https://github.com/josiahls/fastrl.git --depth 1
RUN pip install setuptools==60.7.0
COPY --chown=$CONTAINER_USER:$CONTAINER_GROUP  . fastrl

RUN sudo apt-get -y install cmake

# RUN curl https://get.modular.com | sh - && \
#     modular auth mut_9b52dfea7b05427385fdeddc85dd3a64 && \
#     modular install mojo

# RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastrl/data &&  mv pyproject.toml pyproject.toml_tmp && pip install -e . --no-dependencies &&  mv pyproject.toml_tmp pyproject.toml && cd ../; fi"

RUN /bin/bash -c "if [[ $BUILD == 'prod' ]] ; then echo \"Production Build\" && cd fastrl && pip install . --no-dependencies; fi"
RUN /bin/bash -c "if [[ $BUILD == 'dev' ]] ; then echo \"Development Build\" && cd fastrl && pip install -e \".[dev]\" --no-dependencies ; fi"

# RUN echo '#!/bin/bash\npip install -e .[dev] --no-dependencies && xvfb-run -s "-screen 0 1400x900x24" jupyter lab --ip=0.0.0.0 --port=8080 --allow-root --no-browser  --NotebookApp.token='' --NotebookApp.password=''' >> run_jupyter.sh

RUN /bin/bash -c "cd fastrl && pip install -e . --no-dependencies"

