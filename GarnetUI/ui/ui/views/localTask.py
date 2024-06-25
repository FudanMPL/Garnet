import datetime
import os
import subprocess
from typing import List, Tuple

from django.conf import settings
from django.http import StreamingHttpResponse
from django_q.tasks import async_task
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import OpenApiParameter, extend_schema
from Model.models import DataTaskRelationship, LocalTask
from Model.serializers import (
    DataTaskRelationshipModelSerializer,
    LocalTaskModelSerializer,
    NoneSerializer,
)
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.viewsets import GenericViewSet, ModelViewSet

from ..authentication import UserAuthentication
from ..pagination import PagePagination


class LocalTaskSets(ModelViewSet):
    queryset = LocalTask.objects.all()
    serializer_class = LocalTaskModelSerializer
    authentication_classes = [UserAuthentication]
    pagination_class = PagePagination


class DataTaskRelationshipSets(GenericViewSet):
    queryset = DataTaskRelationship.objects.all()
    serializer_class = DataTaskRelationshipModelSerializer
    authentication_classes = [UserAuthentication]
    pagination_class = PagePagination

    @extend_schema(
        request=DataTaskRelationshipModelSerializer(many=True),
        responses={
            status.HTTP_400_BAD_REQUEST: None,
            status.HTTP_200_OK: DataTaskRelationshipModelSerializer(many=True),
        },
        description="为任务指定数据",
    )
    def create_update(self, request, *args, **kwargs):
        manydata = self.serializer_class(data=request.data, many=True)
        if not manydata.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        manydata.save()
        task = LocalTask.objects.get(id=manydata.data[0]["task"])
        if task.pN == self.queryset.filter(task=task).count():
            task.status = "就绪"
            task.save()
        return Response(data=manydata.data, status=status.HTTP_200_OK)

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="id",
                type=int,
                description="任务id",
                location=OpenApiParameter.PATH,
            )
        ],
        responses={
            status.HTTP_400_BAD_REQUEST: None,
            status.HTTP_200_OK: DataTaskRelationshipModelSerializer(many=True),
        },
        description="查看任务已指定的数据",
    )
    def retrive(self, request, id):
        try:
            task = LocalTask.objects.get(id=id)
        except Exception:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        return Response(
            data=self.serializer_class(
                instance=self.queryset.filter(task=task),
                many=True,
            ).data,
            status=status.HTTP_200_OK,
        )


class LTask:
    task: LocalTask
    mpc: str
    protocol: str
    prefix: str
    pN: int
    data: List[Tuple[int, str]]
    mpc_parameters: str
    protocol_parameters: str
    id: int

    def __init__(self, task: LocalTask) -> None:
        if task is None:
            return
        self.id = task.pk
        self.mpc = os.path.basename(str(task.mpc.file))
        self.protocol = task.protocol.name
        self.prefix = str(task.prefix)
        self.pN = task.pN
        self.data = []
        self.task = task
        if task.mpc_parameters:
            self.mpc_parameters = task.mpc_parameters
        else:
            self.mpc_parameters = ""
        if task.protocol_parameters:
            self.protocol_parameters = task.protocol_parameters
        else:
            self.protocol_parameters = ""
        relationship = DataTaskRelationship.objects.filter(task=task.pk).all()
        for r in relationship:
            self.data.append((r.index, str(r.data.file)))

    def link(self):
        for index, filepath in self.data:
            subprocess.Popen(
                f"ln -s {settings.MEDIA_ROOT}/{filepath} {settings.GARNETPATH}/Input/{self.prefix}-P{index}-0",
                shell=True,
            ).wait()

    def compile(self):
        if self.task.mpc.status == "compiled":
            return
        self.task.mpc.status = "compiled"
        self.task.mpc.save()
        return subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./compile.py {self.mpc_parameters if self.mpc_parameters else ''} {settings.MEDIA_ROOT}/mpc/{self.mpc}",
            shell=True,
        )

    def protocolCompile(self):
        return subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh test -e {self.protocol}.x || {settings.GARNETPATH} make -j 8 {self.protocol}.x",
            shell=True,
        )

    def ssl(self):
        return subprocess.Popen(
            f"{settings.GARNETPATH}/Scripts/setup-ssl.sh {self.pN}",
            shell=True,
            cwd=settings.GARNETPATH,
        )

    def run(self):
        pre = []
        self.link()
        pre.append(self.compile())
        pre.append(self.protocolCompile())
        pre.append(self.ssl())
        for p in pre:
            if p is not None:
                p.wait()
        threads = []
        for i in range(0, self.pN):
            inputPrefix = settings.GARNETPATH + "/Input/" + self.prefix
            outputPrefix = settings.GARNETPATH + "/Output/" + self.prefix
            threads.append(
                subprocess.Popen(
                    f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./{self.protocol}.x {os.path.splitext(self.mpc)[0]} -h localhost -p {i}  -IF {inputPrefix} -pn 7012 -OF  {outputPrefix} { self.protocol_parameters if self.protocol_parameters else ''}",
                    shell=True,
                )
            )
        for t in threads:
            t.wait()
        self.task.status = "已完成"
        self.task.end_time = datetime.datetime.now()
        self.task.save()


class LocalRun(GenericViewSet):
    serializer_class = NoneSerializer
    authentication_classes = [UserAuthentication]

    @extend_schema(
        description="计算本地任务",
        parameters=[
            OpenApiParameter(
                name="taskID",
                type=int,
                location=OpenApiParameter.PATH,
                description="远程任务表主键ID",
            )
        ],
    )
    def run(self, request, taskID: int):
        task = LocalTask.objects.get(id=taskID)
        lt = LTask(task)
        task.status = "运行中"
        task.run_time = datetime.datetime.now()
        task.save()
        print(lt.pN)
        async_task(lt.run)
        return Response({"msg": "运行中"}, status=status.HTTP_200_OK)


class LocalResultsView(APIView):
    authentication_classes = [UserAuthentication]

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="id",
                type=int,
                description="本地任务表主键",
                location=OpenApiParameter.PATH,
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiTypes.BINARY,
            status.HTTP_400_BAD_REQUEST: None,
        },
        description="获取本地计算的结果",
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

        task = LocalTask.objects.get(id=id)
        if not task:
            return Response(data=None, status=status.HTTP_400_BAD_REQUEST)
        filePath = settings.GARNETPATH + "/Output/" + str(task.prefix) + "-P0-0"
        response = StreamingHttpResponse(
            file_iterator(filePath),
            content_type="application/msword",
            status=status.HTTP_200_OK,
        )
        response["Content-Type"] = "application/octet-stream"
        response["Content-Disposition"] = 'attachment;filename="{0}"'.format(
            str(task.prefix) + "-P0-0"
        )
        return response
