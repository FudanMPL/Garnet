FROM python:3.10.3-bullseye as buildenv

RUN apt-get update && apt-get install -y --no-install-recommends \
                automake \
                build-essential \
                clang-11 \
                yasm \
		cmake \
                git \
                libboost-dev \
                libboost-thread-dev \
                libclang-dev \
                libgmp-dev \
                libntl-dev \
                libsodium-dev \
                libssl-dev \
                libtool \
                vim \
                gdb \
                valgrind \
                iproute2 \
                sudo \
        && rm -rf /var/lib/apt/lists/*

ENV Garnet_HOME /usr/src/Garnet
WORKDIR $Garnet_HOME

COPY . .


RUN make clean-deps boost libote

RUN apt-get update && apt-get install -y --no-install-recommends texinfo

RUN make clean
RUN make -j tldr
RUN make -j


RUN pip3 install -r requirements.txt

CMD ["/bin/bash"]