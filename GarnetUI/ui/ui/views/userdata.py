import os
from Model.models import UserData, Users
from Model.serializers import UserDataModelSerializer, NoneSerializer
from rest_framework.viewsets import ModelViewSet, GenericViewSet
from rest_framework.response import Response
from rest_framework import status
from pathlib import Path
from django.conf import settings
from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    inline_serializer,
)
from rest_framework import serializers
from ..authentication import UserAuthentication


class UserdataSets(ModelViewSet):
    queryset = UserData.objects.all()
    serializer_class = UserDataModelSerializer
    authentication_classes = [UserAuthentication]

    def list(self, request, *args, **kwargs):
        user = request.user
        if user.role == 0:
            return super().list(request, *args, **kwargs)
        return Response(
            data=self.serializer_class(
                instance=self.queryset.filter(user=user), many=True
            ).data,
            status=status.HTTP_200_OK,
        )


class UserdataFormStringView(GenericViewSet):
    serializer_class = NoneSerializer
    authentication_classes = [UserAuthentication]

    @extend_schema(
        description="通过字符串上传文件",
        request=inline_serializer(
            name="上传序列化器",
            fields={
                "content": serializers.CharField(),
                "fileName": serializers.CharField(),
                "userID": serializers.IntegerField(),
                "description": serializers.CharField(),
            },
        ),
        responses={
            status.HTTP_200_OK: None,
            status.HTTP_400_BAD_REQUEST: OpenApiResponse(description="错误的用户"),
            status.HTTP_409_CONFLICT: OpenApiResponse(description="文件名重名"),
        },
    )
    def str2file(self, request):
        data = request.data
        path = "data/" + str(data["fileName"])
        try:
            UserData.objects.get(file=path)
            return Response({"msg": "文件名重名"}, status=status.HTTP_409_CONFLICT)
        except:
            pass
        userdata = UserData()
        userdata.user = request.user
        userdata.file.name = path
        userdata.description = data["description"]
        userdata.save()
        path = Path().joinpath(settings.MEDIA_ROOT, path)
        if not os.path.exists(path.parents[0]):
            os.makedirs(path.parents[0])
        with open(file=path, mode="wb") as f:
            f.write(str(data["content"]).encode())
            f.close()
        return Response(status=status.HTTP_200_OK)
