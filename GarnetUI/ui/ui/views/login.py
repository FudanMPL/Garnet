from django.conf import settings
from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiResponse,
    extend_schema,
    inline_serializer,
)
from jose import jwt
from Model import models
from Model.serializers import UserModelSerializer
from rest_framework import serializers, status
from rest_framework.response import Response
from rest_framework.views import APIView
from utils.common import md5


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
        examples=[
            OpenApiExample(
                "Example 1", value={"username": "DSGLAB", "password": "FUDAN"}
            )
        ],
    )
    def post(self, request, *args, **kwargs):
        username = request.data.get("username")
        password = request.data.get("password")
        # 校验用户，因为密码是注册时经过md5加密的，所以这里需要对比加密后的内容
        user_obj = models.Users.objects.filter(
            username=username, password=md5(password)
        ).first()
        if user_obj:
            token = jwt.encode(
                {"username": username},
                settings.SECRET_KEY,
                "HS256",
            )
            return Response(status=status.HTTP_200_OK, data={"token": token})
        else:
            return Response(status=status.HTTP_403_FORBIDDEN)
