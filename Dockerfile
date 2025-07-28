FROM python:3.10.3-slim AS buildenv

RUN apt-get update -o Acquire::Retries=3  && apt-get install -y --no-install-recommends \
                automake \
                build-essential \
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
                valgrind \
                texinfo \
                && rm -rf /var/lib/apt/lists/*

ENV Garnet_HOME=/usr/src/Garnet
WORKDIR $Garnet_HOME

COPY . .

RUN make clean-deps boost libote 
RUN make -j8 tldr && make -j8 semi2k-party.x semi-party.x sml-party.x replicated-ring-party.x shamir-party.x
RUN make clean-intermediate

RUN pip install --no-cache-dir -r requirements.txt


CMD ["/bin/bash"]