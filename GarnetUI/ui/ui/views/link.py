from pathlib import Path
from rest_framework.response import Response
from rest_framework import status
from rest_framework.viewsets import GenericViewSet
from rest_framework.mixins import ListModelMixin
from drf_spectacular.utils import extend_schema
from django_q.tasks import async_task
from django.conf import settings
from Model.models import Servers, RemoteTask, ServerTaskRelationship, Protocol, Mpc
from Model.serializers import (
    ServersModelSerializer,
    MetadataSerializer,
    TaskResponseSerializer,
    TaskRequestSerializer,
    JoinedServersSerializer,
    RemoteTaskModelSerializer,
    NoneSerializer,
)
import utils.common, requests, subprocess, uuid
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse
from utils.common import md5
from uuid import uuid4


class MetadataView(GenericViewSet):
    """
    逻辑是这样的：用户发送一个MetadataSerializer，当中指明了这台服务器需要连接的另一台服务器的ip与port。
    该服务器向该ip、port发送一个ServersModelSerializer，当中是本机的元数据。远端服务器接收后，返回自己的ServersModelSerializer，双方完成同步。
    """

    serializer_class = NoneSerializer

    @extend_schema(
        request=ServersModelSerializer,
        responses={status.HTTP_202_ACCEPTED: Servers},
        description="服务器连接，接收方",
    )
    def receive(self, request, *args, **kwargs):
        localMetadata = ServersModelSerializer(instance=metadataUpdate())
        [server, _] = Servers.objects.get_or_create(
            servername=request.data["servername"],
            ip=request.data["ip"],
            port=request.data["port"],
            token=request.data["token"],
        )
        path = Path.joinpath(
            settings.MEDIA_ROOT, "ssl", request.data["servername"] + ".pem"
        )
        async_task(utils.common.download, request.data["url"], path)
        server.save()

        return Response(data=localMetadata.data, status=status.HTTP_200_OK)

    @extend_schema(
        request=MetadataSerializer,
        responses={status.HTTP_202_ACCEPTED: Servers},
        description="服务器连接，发送方",
    )
    def send(self, request):
        hostSerializer = MetadataSerializer(instance=None, data=request.data)
        if not hostSerializer.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        try:
            data = link_ssl(
                hostSerializer.data["ip"],
                int(hostSerializer.data["port"]),
            )
        except Exception as err:
            return Response(status=err)
        return Response(
            data=ServersModelSerializer(instance=data).data,
            status=status.HTTP_202_ACCEPTED,
        )


class TaskReleaseView(GenericViewSet, ListModelMixin):
    queryset = RemoteTask.objects.all()
    serializer_class = TaskResponseSerializer

    @extend_schema(
        request=TaskRequestSerializer,
        responses={
            status.HTTP_200_OK: TaskResponseSerializer,
            status.HTTP_400_BAD_REQUEST: None,
            status.HTTP_409_CONFLICT: None,
        },
        description="发布任务",
    )
    def release(self, request):
        r = TaskRequestSerializer(data=request.data)
        if not r.is_valid():
            return Response(status=status.HTTP_400_BAD_REQUEST)
        prefix = r.data["prefix"]
        servername = r.data["servername"]
        part = r.data["part"]
        try:
            server = Servers.objects.get(servername=servername)
        except Servers.DoesNotExist:
            return Response(data=None, status=status.HTTP_409_CONFLICT)
        try:
            task = RemoteTask.objects.get(prefix=prefix)

            if (
                ServerTaskRelationship.objects.filter(part=part, task=task)
                or part > task.pN
                or task.status != "等待参与方加入"
            ):
                return Response(data=None, status=status.HTTP_409_CONFLICT)

            relationship = ServerTaskRelationship()
            relationship.part = part
            relationship.task = task
            relationship.server = server
            relationship.save()
            if ServerTaskRelationship.objects.filter(task=task).count() >= task.pN:
                if task.data != None:
                    task.status = "本地就绪"
                else:
                    task.status = "等待数据"
                task.save()
            return Response(
                data=TaskResponseSerializer(instance=task).data,
                status=status.HTTP_200_OK,
            )
        except RemoteTask.DoesNotExist:
            return Response(data=None, status=status.HTTP_400_BAD_REQUEST)

    @extend_schema(
        description="向指定任务的各方服务器发送各自的服务器元数据",
        parameters=[
            OpenApiParameter(
                name="taskID",
                type=int,
                location=OpenApiParameter.PATH,
                description="远程任务ID",
            )
        ],
        responses={status.HTTP_200_OK: None},
    )
    def serverSend(self, request, taskID: int):
        servers = ServerTaskRelationship.objects.filter(task=taskID)
        joined = JoinedServersSerializer(instance=servers, many=True)
        data = list(joined.data)
        data.append(str(RemoteTask.objects.get(id=taskID).prefix))
        for server in servers:
            requests.post(
                f"http://{server.server.ip}:{server.server.port}/api/link/server/receive/",
                json=data,
            )

        return Response(status=status.HTTP_200_OK)


class TaskJoinView(GenericViewSet):
    serializer_class = NoneSerializer

    @extend_schema(
        parameters=[
            OpenApiParameter(
                name="serverID",
                type=int,
                description="联络的服务器",
                location=OpenApiParameter.PATH,
            ),
            OpenApiParameter(
                name="prefix",
                type=uuid.UUID,
                description="任务的UUID",
                location=OpenApiParameter.PATH,
            ),
            OpenApiParameter(
                name="part",
                type=int,
                description="自己想作为第几方",
                location=OpenApiParameter.PATH,
            ),
        ],
        responses={
            status.HTTP_200_OK: TaskResponseSerializer,
            status.HTTP_400_BAD_REQUEST: None,
            status.HTTP_409_CONFLICT: None,
        },
        description="参与一个远程计算任务",
    )
    def join(self, request, serverID: int, prefix: uuid.UUID, part: int):
        try:
            server = Servers.objects.get(id=serverID)
        except Servers.DoesNotExist:
            return Response(status=status.HTTP_400_BAD_REQUEST)
        req = TaskRequestSerializer(
            {"servername": settings.NAME, "prefix": prefix, "part": part}
        )

        res = requests.post(
            "http://"
            + server.ip
            + ":"
            + str(server.port)
            + "/api/task/remote/release/",
            data=req.data,
        )
        if res.status_code != 200:
            return Response(status=res.status_code)

        dic = res.json()
        dic["part"] = part
        dic["protocol"] = Protocol.objects.get(name=res.json()["protocolName"])
        name = res.json()["mpcURL"].split("/")[-1].split(".")[0]

        [mpc, _] = Mpc.objects.get_or_create(name=name, description=dic["description"])
        mpc.file.name = "mpc" + "/" + name + ".mpc"
        path = Path.joinpath(settings.MEDIA_ROOT, "mpc", mpc.file.name)
        async_task(utils.common.download, res.json()["mpcURL"], path)
        mpc.save()

        dic["mpc"] = mpc
        dic["status"] = "等待数据"
        dic.pop("mpcURL")
        dic.pop("protocolName")
        dic.pop("id")
        task = RemoteTask(**dic)
        task.save()
        s = RemoteTaskModelSerializer(instance=task)
        return Response(data=s.data, status=status.HTTP_200_OK)

    @extend_schema(description="接受参与指定任务的各方服务器元数据", request=JoinedServersSerializer)
    def serverReceive(self, request):
        prefix = request.data[-1]
        task = RemoteTask.objects.get(prefix=prefix)
        del request.data[-1]
        joined = request.data
        delete = set(ServerTaskRelationship.objects.filter(task=task.pk))
        for s in joined:
            try:
                server = link_ssl(s["serverIP"], s["serverPort"])
            except Exception as err:
                continue
            try:
                relationship = ServerTaskRelationship.objects.get(
                    task=task.pk, server=server.pk
                )
                delete.remove(relationship)
            except:
                relationship = ServerTaskRelationship()
                relationship.server = server
                relationship.task = task
                relationship.part = s["part"]
                relationship.save()
        for d in delete:
            d.delete()
        return Response(data=request.data, status=status.HTTP_200_OK)


class ReadyView(GenericViewSet):
    serializer_class = NoneSerializer

    @extend_schema(
        description="询问各方是否已就绪",
        request=[
            OpenApiParameter(
                name="prefix",
                type=uuid.UUID,
                location=OpenApiParameter.PATH,
                description="任务前缀",
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiResponse(description="已就绪"),
            status.HTTP_204_NO_CONTENT: OpenApiResponse(description="本地数据未就绪"),
            status.HTTP_400_BAD_REQUEST: OpenApiResponse(description="出错"),
            status.HTTP_425_TOO_EARLY: OpenApiResponse(description="其余服务器就绪"),
        },
    )
    def ready(self, request, prefix: uuid.UUID):
        task = RemoteTask.objects.get(prefix=prefix)
        if task.status != "就绪":
            return Response(
                {"msg": "not ready localy"}, status=status.HTTP_204_NO_CONTENT
            )
        servers = ServerTaskRelationship.objects.filter(task=task).exclude(server=1)
        ready = True
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
        if ready:
            task.status = "就绪"
            task.save()
            return Response({"msg": "就绪"}, status=status.HTTP_200_OK)
        return Response({"msg": "not ready"}, status=status.HTTP_425_TOO_EARLY)

    @extend_schema(
        description="询问本方是否已就绪",
        request=[
            OpenApiParameter(
                name="prefix",
                type=uuid.UUID,
                location=OpenApiParameter.PATH,
                description="任务前缀",
            )
        ],
        responses={
            status.HTTP_200_OK: OpenApiResponse(description="已就绪"),
            status.HTTP_204_NO_CONTENT: OpenApiResponse(description="需要更新服务器元数据"),
            status.HTTP_425_TOO_EARLY: OpenApiResponse(description="数据未就绪"),
            status.HTTP_400_BAD_REQUEST: OpenApiResponse(description="任务不存在"),
        },
    )
    def isReady(self, request, prefix: uuid.UUID):
        try:
            task = RemoteTask.objects.get(prefix=prefix)
        except RemoteTask.DoesNotExist:
            return Response(
                {"msg": "task doesn't exist"}, status=status.HTTP_400_BAD_REQUEST
            )
        if ServerTaskRelationship.objects.filter(task=task).count() < task.pN:
            return Response(
                {"msg": "servers need update"}, status=status.HTTP_204_NO_CONTENT
            )
        if task.data == None:
            return Response({"msg": "not ready"}, status=status.HTTP_425_TOO_EARLY)
        return Response({"msg": "就绪"}, status=status.HTTP_200_OK)


def metadataUpdate():
    try:
        server = Servers.objects.get(id=1)
        server.ip = settings.IPADDRESS
        server.port = settings.PORT
        if server.servername != settings.NAME:
            subprocess.Popen(
                f"mv {settings.BASE_DIR}/uploads/ssl/{server.servername}.pem {settings.BASE_DIR}/uploads/ssl/{settings.NAME}.pem",
                shell=True,
            )
            subprocess.Popen(
                f"mv {settings.BASE_DIR}/uploads/ssl/{server.servername}.key {settings.BASE_DIR}/uploads/ssl/{settings.NAME}.key",
                shell=True,
            )
            server.servername = settings.NAME
            server.save()
    except Servers.DoesNotExist:
        subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/ssl.sh {settings.NAME} {settings.BASE_DIR}/uploads/ssl",
            shell=True,
        )
        server = Servers.objects.create(
            id=1, servername=settings.NAME, ip=settings.IPADDRESS, port=settings.PORT
        )
        server.token = md5(server.servername + str(uuid4()))
        server.save()
    return server


def link_ssl(host: str, port: int) -> Servers:
    """
    主动建立连接并交换ssl密钥
    """
    m = ServersModelSerializer(instance=metadataUpdate())
    res = requests.post(
        "http://" + host + ":" + str(port) + "/api/link/metadata/receive/",
        data=m.data,
    )
    if res.status_code != 200:
        raise Exception(res.status_code)
    try:
        server = Servers.objects.get(servername=res.json()["servername"])
        server.ip = res.json()["ip"]
        server.port = port
        server.token = res.json()["token"]
        server.save()
    except Servers.DoesNotExist:
        server = Servers()
        server.ip = res.json()["ip"]
        server.servername = res.json()["servername"]
        server.port = port
        server.token = res.json()["token"]
        path = Path.joinpath(
            settings.MEDIA_ROOT, "ssl", res.json()["servername"] + ".pem"
        )
        async_task(utils.common.download, res.json()["url"], path)
        server.save()
    return server
