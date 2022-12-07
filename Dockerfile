FROM tensorflow/tensorflow:2.10.0-gpu
ENV DEBIAN_FRONTEND noninteractive
RUN mkdir /dev_env
WORKDIR /dev_env
COPY . .
RUN apt-get update && apt-get upgrade -y && apt-get install -y
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y cmake libopenmpi-dev zlib1g-dev
RUN apt-get install -y libcairo2-dev
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
RUN apt-get install -y git
RUN apt-get install -y vim-nox
RUN apt-get install -y tree
RUN apt-get install -y tzdata
RUN apt-get install -y openssh-server
RUN sed -ri 's/PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config
RUN sed -ri 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN apt-get autoremove -y
RUN python3 -m pip install --upgrade pip
RUN pip3 install --upgrade tf_slim
RUN pip3 install gym==0.11.0
RUN pip3 install --no-cache-dir -r /dev_env/requirements.txt
RUN pip3 install --no-cache-dir -r /dev_env/ci_requirements.txt
RUN passwd -d root
RUN echo "service ssh start" >> /root/.bashrc
EXPOSE 22


