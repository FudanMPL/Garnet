import uuid
from django.conf import settings
import datetime
import os
import subprocess
from typing import Dict

import requests
from django.http import StreamingHttpResponse
from drf_spectacular.types import OpenApiTypes
from django_q.tasks import async_task
from Model.models import RemoteTask, ServerTaskRelationship, Servers, UserData
from Model.serializers import (
    NoneSerializer,
    RemoteTaskModelSerializer,
    ServerTaskRelationshipModelSerializer,
)
from rest_framework.mixins import UpdateModelMixin
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.response import Response
from rest_framework import status
from .link import metadataUpdate, TaskReleaseView
from drf_spectacular.utils import (
    extend_schema,
    OpenApiExample,
    inline_serializer,
    OpenApiParameter,
    OpenApiResponse,
)
from rest_framework import serializers


class RemoteTaskSets(ModelViewSet):
    queryset = RemoteTask.objects.all()
    serializer_class = RemoteTaskModelSerializer

    @extend_schema(
        description="创建远程任务表",
        request=RemoteTaskModelSerializer,
        responses={
            status.HTTP_201_CREATED: RemoteTaskModelSerializer,
            status.HTTP_400_BAD_REQUEST: None,
        },
    )
    def create(self, request, *args, **kwargs):
        relationship = ServerTaskRelationship()
        s = self.serializer_class(data=request.data)
        if not s.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        obj = s.save()
        relationship.task = obj
        relationship.part = 0
        try:
            relationship.server = Servers.objects.get(id=1)
        except Servers.DoesNotExist:
            m = metadataUpdate()
            relationship.server = m
        relationship.save()
        return Response(status=status.HTTP_200_OK)

    def release(self, request):
        tasks = RemoteTask.objects.filter(status="等待参与方加入")
        return Response(
            status=status.HTTP_200_OK,
            data=RemoteTaskModelSerializer(instance=tasks, many=True).data,
        )


class ServerTaskRelationshipSets(GenericViewSet, UpdateModelMixin):
    queryset = ServerTaskRelationship.objects.all()
    serializer_class = ServerTaskRelationshipModelSerializer


class RemoteTaskAddData(GenericViewSet, UpdateModelMixin):
    queryset = RemoteTask.objects.all()
    serializer_class = RemoteTaskModelSerializer

    @extend_schema(
        description="更新远程任务表的数据",
        parameters=[
            OpenApiParameter(
                name="id",
                type=int,
                location=OpenApiParameter.PATH,
                description="要插入数据的任务id",
            )
        ],
        request=inline_serializer(
            name="更新序列化器",
            fields={
                "data": serializers.IntegerField(),
            },
        ),
        responses={
            status.HTTP_200_OK: RemoteTaskModelSerializer,
            status.HTTP_400_BAD_REQUEST: None,
        },
    )
    def partial_update(self, request, *args, **kwargs):
        task = self.get_object()
        try:
            data = UserData.objects.get(id=request.data["data"])
        except UserData.DoesNotExist:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        task.data = data
        if task.host == settings.IPADDRESS:
            if ServerTaskRelationship.objects.filter(task=task).count() == task.pN:
                task.status = "本地就绪"
        else:
            task.status = "就绪"
        task.save()
        return Response(
            RemoteTaskModelSerializer(instance=task).data, status=status.HTTP_200_OK
        )


class RemoteResultsView(GenericViewSet):
    serializer_class = NoneSerializer

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="id",
                type=int,
                description="远程任务表主键",
                location=OpenApiParameter.PATH,
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiTypes.BINARY,
            status.HTTP_400_BAD_REQUEST: None,
        },
        description="获取远程计算的结果",
    )
    def get(self, request, id: int):
        def file_iterator(file_name, chunk_size=512):
            with open(file_name, "rb") as f:
                while True:
                    c = f.read(chunk_size)
                    if c:
                        yield c
                    else:
                        break

        task = RemoteTask.objects.get(id=id)
        if not task:
            return Response(data=None, status=status.HTTP_400_BAD_REQUEST)
        filePath = (
            settings.GARNETPATH
            + "/Output/"
            + str(task.prefix)
            + "-P"
            + str(task.part)
            + "-0"
        )
        response = StreamingHttpResponse(
            file_iterator(filePath),
            content_type="application/msword",
            status=status.HTTP_200_OK,
        )
        response["Content-Type"] = "application/octet-stream"
        response["Content-Disposition"] = 'attachment;filename="{0}"'.format(
            str(task.prefix) + "-P" + str(task.pN) + "-0"
        )
        return response


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
        if task == None or task.data == None:
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
        self.data = str(task.data.file.name)
        self.mpc_parameters = str(task.mpc_parameters)
        self.protocol_parameters = str(task.protocol_parameters)
        self.servers = {}
        self.serverName = Servers.objects.get(id=1).servername
        servers = ServerTaskRelationship.objects.filter(task=task).exclude(server=1)
        for server in servers:
            self.servers[server.part] = server.server.servername

    def rehash(self):
        s = ""
        for k, v in self.servers.items():
            s = f"{s} {k} {v}"
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/rehash.sh {settings.MEDIA_ROOT}/ssl {settings.GARNETPATH} {self.part} {self.serverName} {s}",
            shell=True,
        ).wait()

    def compile(self):
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./compile.py {self.mpc_parameters if self.mpc_parameters else ''} {settings.MEDIA_ROOT}/mpc/{self.mpc}",
            shell=True,
        ).wait()

    def link(self):
        subprocess.Popen(
            f"ln -s {settings.MEDIA_ROOT}/{self.data} {settings.GARNETPATH}/Input/{self.prefix}-P{self.part}-0",
            shell=True,
        ).wait()

    def run(self):
        self.rehash()
        self.compile()
        self.link()
        inputPrefix = settings.GARNETPATH + "/Input/" + self.prefix
        outputPrefix = settings.GARNETPATH + "/Output/" + self.prefix
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./{self.protocol}.x {os.path.splitext(self.mpc)[0]} -h {self.host} -pn {self.basePort} -p {self.part} -IF {inputPrefix} -OF {outputPrefix} ",
            shell=True,
        ).wait()
        self.task.status = "已完成"
        self.task.end_time = datetime.datetime.now()
        self.task.save()


class RemoteRun(GenericViewSet):
    serializer_class = NoneSerializer

    @extend_schema(
        description="协调方发布计算命令",
        parameters=[
            OpenApiParameter(
                name="taskID",
                type=int,
                location=OpenApiParameter.PATH,
                description="远程任务表主键ID",
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiResponse(description="运行中"),
            status.HTTP_204_NO_CONTENT: OpenApiResponse(description="本方未就绪"),
            status.HTTP_400_BAD_REQUEST: OpenApiResponse(description="本方不是协调方"),
            status.HTTP_425_TOO_EARLY: OpenApiResponse(description="其余方未就绪"),
        },
    )
    def coordinator(self, request, taskID: int):
        task = RemoteTask.objects.get(id=taskID)
        if task.host != settings.IPADDRESS:
            return Response(
                {"msg": "本方不是协调方，不能发布计算命令"}, status=status.HTTP_400_BAD_REQUEST
            )
        if task.status != "本地就绪" and task.status != "就绪":
            return Response({"msg": "本方未就绪"}, status=status.HTTP_204_NO_CONTENT)
        servers = ServerTaskRelationship.objects.filter(task=task).exclude(server=1)
        ready = True
        if task.status == "本地就绪":
            for server in servers:
                url = f"http://{server.server.ip}:{server.server.port}/api/link/ready/receive/{task.prefix}"
                r = requests.get(url=url)
                match r.status_code:
                    case status.HTTP_400_BAD_REQUEST:
                        return Response(
                            {"msg": "relationship error"},
                            status=status.HTTP_400_BAD_REQUEST,
                        )
                    case status.HTTP_204_NO_CONTENT:
                        ready = False
                        t = TaskReleaseView()
                        t.serverSend(None, task.pk)
                    case status.HTTP_425_TOO_EARLY | status.HTTP_408_REQUEST_TIMEOUT | status.HTTP_404_NOT_FOUND | status.HTTP_500_INTERNAL_SERVER_ERROR:
                        ready = False
            if not ready:
                return Response({"msg": "其余各方未就绪"}, status=status.HTTP_425_TOO_EARLY)
        for server in servers:
            url = f"http://{server.server.ip}:{server.server.port}/api/task/remote/run/{task.prefix}"
            requests.get(url=url)
        rt = RTask(task)
        task.status = "运行中"
        task.run_time = datetime.datetime.now()
        task.save()
        async_task(rt.run)
        return Response({"msg": "运行中"}, status=status.HTTP_200_OK)

    @extend_schema(
        description="参与方计算",
        parameters=[
            OpenApiParameter(
                name="prefix",
                type=uuid.UUID,
                location=OpenApiParameter.PATH,
                description="远程任务表的prefix",
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiResponse(description="运行中"),
            status.HTTP_425_TOO_EARLY: OpenApiResponse(description="未就绪"),
        },
    )
    def participant(self, request, prefix: uuid.UUID):
        task = RemoteTask.objects.get(prefix=prefix)
        if task.status != "就绪":
            return Response({"msg": "not ready"}, status=status.HTTP_425_TOO_EARLY)
        rt = RTask(task)
        task.status = "运行中"
        task.run_time = datetime.datetime.now()
        task.save()
        async_task(rt.run)
        return Response({"msg": "运行中"}, status=status.HTTP_200_OK)
