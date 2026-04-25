# Systemd-enabled Ubuntu container for sysadmin scenarios
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV container=docker

# Install systemd and common tools needed for scenarios
RUN apt-get update && apt-get install -y \
    systemd \
    systemd-sysv \
    nginx \
    apache2 \
    python3 \
    python3-venv \
    curl \
    netcat-openbsd \
    procps \
    iproute2 \
    openssl \
    coreutils \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /lib/systemd/system/multi-user.target.wants/* \
    && rm -f /etc/systemd/system/*.wants/* \
    && rm -f /lib/systemd/system/local-fs.target.wants/* \
    && rm -f /lib/systemd/system/sockets.target.wants/*udev* \
    && rm -f /lib/systemd/system/sockets.target.wants/*initctl* \
    && rm -f /lib/systemd/system/basic.target.wants/* \
    && rm -f /lib/systemd/system/anaconda.target.wants/*

# Disable services that cause issues in containers
RUN systemctl mask \
    systemd-firstboot.service \
    systemd-udevd.service \
    systemd-modules-load.service \
    sys-kernel-debug.mount \
    sys-kernel-tracing.mount

VOLUME ["/sys/fs/cgroup"]

STOPSIGNAL SIGRTMIN+3

CMD ["/lib/systemd/systemd"]
