FROM tensorflow/tensorflow:2.10.0-gpu
ENV DEBIAN_FRONTEND noninteractive
RUN mkdir /dev_env
WORKDIR /dev_env
COPY . .
RUN cp .bashrc /root/

RUN echo root:admin | chpasswd
RUN apt-get -q update && apt-get upgrade -y \
&& apt-get install -y --no-install-recommends apt-utils \
&& apt-get install -y cmake libopenmpi-dev zlib1g-dev libcairo2-dev \
git vim-nox tree openssh-server tzdata \
&& apt-get autoremove -y

RUN python3 -m pip install --upgrade pip \
&& pip3 install --upgrade tf_slim \
&& pip3 install gym==0.11.0 \
&& pip3 install --no-cache-dir -r /dev_env/requirements.txt \
&& pip3 install --no-cache-dir -r /dev_env/ci_requirements.txt

RUN sed -ri 's/PermitEmptyPasswords no/PermitEmptyPasswords yes/' /etc/ssh/sshd_config \
&& sed -ri 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
&& sed -ri 's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config
RUN mkdir -p /var/run/sshd && chmod 755 /var/run/sshd && chmod 600 -R /etc/ssh

EXPOSE 22 6006
CMD ["service", "ssh", "start"]
