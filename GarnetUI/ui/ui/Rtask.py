from pathlib import Path
from Model.models import RemoteTask, Mpc
from typing import Dict
import os
from Model.models import (
    RemoteTask,
    ServerTaskRelationship,
    ServerTaskRelationship,
    Servers,
)
import subprocess
from django.conf import settings
import datetime
from utils.common import download


class RTask:
    task: RemoteTask
    serverName: str
    part: int
    pN: int
    mpc: str
    protocol: str
    host: str
    basePort: int
    prefix: str
    data: str
    mpc_parameters: str
    protocol_parameters: str
    servers: Dict[int, str]

    def __init__(self, task: RemoteTask) -> None:
        if task == None:
            return
        self.task = task
        self.part = task.part
        self.pN = task.pN
        self.mpc = os.path.basename(str(task.mpc.file))
        self.protocol = task.protocol.name
        self.prefix = str(task.prefix)
        # 这个判断并不会执行
        if task.host == None:
            return
        self.host = task.host
        self.basePort = task.baseport
        if task.data != None:
            self.data = str(task.data.file.name)
        self.mpc_parameters = str(task.mpc_parameters)
        self.protocol_parameters = str(task.protocol_parameters)
        self.servers = {}
        self.serverName = Servers.objects.get(id=1).servername
        servers = ServerTaskRelationship.objects.filter(task=task)
        for server in servers:
            self.servers[server.part] = server.server.servername

    def ssl(self) -> None:
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/ssl.sh {settings.NAME} {settings.BASE_DIR}/uploads/ssl {self.part}",
            shell=True,
        ).wait()

    def rehash(self):
        s = ""
        for k, v in self.servers.items():
            s = f"{s} {k} {v}"
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/rehash.sh {settings.MEDIA_ROOT}/ssl {settings.GARNETPATH} {self.part} {self.serverName} {s}",
            shell=True,
        ).wait()

    def link(self):
        subprocess.Popen(
            f"mkdir {settings.GARNETPATH}/Input & mkdir {settings.GARNETPATH}/Output & ln -s '{settings.MEDIA_ROOT}/{self.data}' '{settings.GARNETPATH}/Input/{self.prefix}-P{self.part}-0'",
            shell=True,
        ).wait()

    def run(self):
        self.rehash()
        self.link()
        inputPrefix = settings.GARNETPATH + "/Input/" + self.prefix
        outputPrefix = settings.GARNETPATH + "/Output/" + self.prefix
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./{self.protocol}.x {os.path.splitext(self.mpc)[0]} -h {self.host} -pn {self.basePort} -p {self.part} -IF {inputPrefix} -OF {outputPrefix}",
            shell=True,
        ).wait()
        self.task.status = "已完成"
        self.task.end_time = datetime.datetime.now()
        self.task.save()


def downloadAndCompile(task: RemoteTask, url: str, path: Path):
    download(url, path)
    mpc_compile(task.mpc, task.mpc_parameters if task.mpc_parameters else "")
    protocol_compile(task.protocol.name)


def mpc_compile(mpc: Mpc, mpc_parameters: str):
    if mpc.status == "compiled":
        return
    mpc.status = "compiling"
    mpc.save()
    subprocess.Popen(
        f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./compile.py {mpc_parameters} {settings.MEDIA_ROOT}/{mpc.file.name}",
        shell=True,
    ).wait()
    mpc.status = "compiled"
    mpc.save()


def protocol_compile(protocol: str):
    subprocess.Popen(
        f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} make {protocol}.x -j8",
        shell=True,
    ).wait()
