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
        && rm -rf /var/lib/apt/lists/*

ENV Ents_HOME /usr/src/Ents
WORKDIR $Ents_HOME

COPY . .


RUN make clean-deps boost libote

RUN make clean
RUN make -j 8 tldr
RUN make -j 8 replicated-ring-party.x
RUN make -j 8 semi2k-party.x
RUN make -j 8 Fake-Offline.x
RUN make -j 8 malicious-rep-ring-party.x




RUN pip3 install -r requirements.txt

RUN ./Scripts/setup-ssl.sh 3



RUN ./Scripts/setup-online.sh 3 32
RUN ./Scripts/setup-online.sh 3 128
RUN ./Scripts/setup-online.sh 2 32
RUN ./Scripts/setup-online.sh 2 128


