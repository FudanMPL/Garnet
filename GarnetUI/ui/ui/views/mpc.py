import os
from Model.models import Mpc
from Model.serializers import MpcModelSerializer, NoneSerializer
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from ..authentication import UserAuthentication
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    inline_serializer,
)
from rest_framework import serializers
from rest_framework import status
from rest_framework.response import Response
from pathlib import Path
from django.conf import settings
import subprocess
from ..pagination import PagePagination


class MpcSets(ModelViewSet):
    authentication_classes = [UserAuthentication]
    queryset = Mpc.objects.all()
    serializer_class = MpcModelSerializer
    pagination_class = PagePagination

class MpcFormStringView(GenericViewSet):
    serializer_class = NoneSerializer
    authentication_classes = [UserAuthentication]

    @extend_schema(
        description="通过字符串上传Mpc文件",
        request=inline_serializer(
            name="上传序列化器",
            fields={
                "content": serializers.CharField(),
                "fileName": serializers.CharField(),
                "description": serializers.CharField(),
            },
        ),
        responses={
            status.HTTP_200_OK: MpcModelSerializer,
            status.HTTP_400_BAD_REQUEST: OpenApiResponse(description="错误的用户"),
            status.HTTP_409_CONFLICT: OpenApiResponse(description="文件名重名"),
        },
    )
    def str2file(self, request):
        data = request.data
        path = "mpc/" + str(data["fileName"])
        try:
            Mpc.objects.get(file=path)
            return Response({"msg": "文件名重名"}, status=status.HTTP_409_CONFLICT)
        except:
            pass
        mpc = Mpc()
        mpc.file.name = path
        mpc.name = mpc.file.name.split("/")[1]
        mpc.description = data["description"]
        mpc.save()
        path = Path().joinpath(settings.MEDIA_ROOT, path)
        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        with open(file=path, mode="wb") as f:
            f.write(str(data["content"]).encode())
            f.close()
        return Response(
            status=status.HTTP_200_OK, data=MpcModelSerializer(instance=mpc).data
        )

    @extend_schema(
        description="试编译mpc文件",
        request=inline_serializer(
            name="编译命令序列化器",
            fields={
                "content": serializers.CharField(),
                "parameters": serializers.CharField(),
            },
        ),
        responses={
            status.HTTP_200_OK: None,
        },
    )
    def tryCompile(self, request):
        data = request.data
        mpc_parameters = data["parameters"]
        path = Path().joinpath(settings.MEDIA_ROOT, "mpc/temp.mpc")
        with open(file=path, mode="wb") as f:
            f.write(str(data["content"]).encode())
            f.close()
        ex = subprocess.Popen(
            f"{settings.BASE_DIR}/scripts/run.sh {settings.GARNETPATH} ./compile.py {mpc_parameters if mpc_parameters else ''} {settings.MEDIA_ROOT}/mpc/temp.mpc",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = ex.communicate()
        subprocess.Popen(
            f"rm {settings.MEDIA_ROOT}/mpc/temp.mpc",
            shell=True,
        )
        return Response(
            status=status.HTTP_200_OK,
            data={"out": out.decode("utf-8"), "err": err.decode("utf-8")},
        )
