from django.conf import settings
from jose import jwt
from Model import models
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed



class UserAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 从header获取token信息
        token = request.headers.get("token")
        if not token:
            raise AuthenticationFailed(
                {"code": "403", "data": None, "msg": "缺少token"}
            )
        username = jwt.decode(
            token,
            settings.SECRET_KEY,
            "HS256",
        )["username"]
        user = models.Users.objects.filter(username=username)
        if user:
            return user, token
        else:
            # 抛出异常
            raise AuthenticationFailed(
                {"code": "401", "data": None, "msg": "token已失效"}
            )

    def authenticate_header(self, request):
        return "token"


class ServerAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 从header获取token信息
        token = request.headers.get("token")
        if not token:
            raise AuthenticationFailed(
                {"code": "403", "data": None, "msg": "缺少token"}
            )
        token_obj = models.Servers.objects.filter(token=token).first()
        if token_obj:
            return token_obj, token
        else:
            # 抛出异常
            raise AuthenticationFailed(
                {"code": "401", "data": None, "msg": "token已失效"}
            )

    def authenticate_header(self, request):
        return "token"
