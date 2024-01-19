import uuid
from django.conf import settings
import datetime

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
    inline_serializer,
    OpenApiParameter,
    OpenApiResponse,
)
from rest_framework import serializers
from ..pagination import PagePagination
from ..Rtask import RTask, mpc_compile, protocol_compile


class RemoteTaskSets(ModelViewSet):
    queryset = RemoteTask.objects.all()
    serializer_class = RemoteTaskModelSerializer
    pagination_class = PagePagination

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
        s = RemoteTaskModelSerializer(data=request.data)
        if not s.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        obj = s.save()
        # 断言，不加这个就报错，尽管不会出问题
        assert isinstance(obj, RemoteTask)
        relationship.task = obj
        relationship.part = 0
        try:
            relationship.server = Servers.objects.get(id=1)
        except Servers.DoesNotExist:
            m = metadataUpdate()
            relationship.server = m
        relationship.save()
        async_task(mpc_compile, obj.mpc, obj.mpc_parameters)
        async_task(protocol_compile, obj.protocol.name)
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
        if task.part == 0:
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
