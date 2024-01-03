from uuid import uuid4
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status, serializers

from Model import models
from Model.serializers import UserModelSerializer

from utils.common import md5, set_expires_time

from drf_spectacular.utils import (
    extend_schema,
    OpenApiResponse,
    inline_serializer,
)


class LoginView(APIView):
    @extend_schema(
        request=UserModelSerializer,
        responses={
            status.HTTP_200_OK: OpenApiResponse(
                response=inline_serializer(
                    name="token", fields={"token": serializers.CharField()}
                ),
                description="返回token",
            ),
            status.HTTP_403_FORBIDDEN: OpenApiResponse(description="登陆错误"),
        },
    )
    def post(self, request, *args, **kwargs):
        username = request.data.get("username")
        password = request.data.get("password")
        # 校验用户，因为密码是注册时经过md5加密的，所以这里需要对比加密后的内容
        user_obj = models.Users.objects.filter(
            username=username, password=md5(password)
        ).first()
        if user_obj:
            # 生成token，存储到数据库
            token = md5(username + str(uuid4()))
            models.Token.objects.update_or_create(
                user=user_obj,
                defaults={"token": token, "expires_time": set_expires_time()},
            )
            return Response(status=status.HTTP_200_OK, data={"token": token})
        else:
            return Response(status=status.HTTP_403_FORBIDDEN)
