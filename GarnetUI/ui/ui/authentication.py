from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed
from Model import models
from utils.common import is_expiration


class UserAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 从header获取token信息
        token = request.headers.get("token")
        if not token:
            raise AuthenticationFailed({"code": "403", "data": None, "msg": "缺少token"})
        token_obj = models.Token.objects.filter(token=token).first()
        # 校验token有效期
        if token_obj and is_expiration(token_obj.expires_time):
            return token_obj.user, token
        else:
            # 抛出异常
            raise AuthenticationFailed({"code": "401", "data": None, "msg": "token已失效"})

    def authenticate_header(self, request):
        return "token"


class ServerAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 从header获取token信息
        token = request.headers.get("token")
        if not token:
            raise AuthenticationFailed({"code": "403", "data": None, "msg": "缺少token"})
        token_obj = models.Servers.objects.filter(token=token).first()
        if token_obj:
            return token_obj, token
        else:
            # 抛出异常
            raise AuthenticationFailed({"code": "401", "data": None, "msg": "token已失效"})

    def authenticate_header(self, request):
        return "token"
