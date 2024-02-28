from Model.models import Users
from Model.serializers import UserModelSerializer
from drf_spectacular.utils import extend_schema, OpenApiExample
from rest_framework.viewsets import GenericViewSet
from rest_framework.mixins import (
    UpdateModelMixin,
    CreateModelMixin,
    ListModelMixin,
    DestroyModelMixin,
)
from rest_framework import status
from ..authentication import UserAuthentication
from ..pagination import PagePagination


class UserSets(GenericViewSet, ListModelMixin, DestroyModelMixin, UpdateModelMixin):
    queryset = Users.objects
    serializer_class = UserModelSerializer
    lookup_field = "username"
    authentication_classes = [UserAuthentication]
    pagination_class = PagePagination

    @extend_schema(
        request=None,
        responses={status.HTTP_200_OK: UserModelSerializer(many=True)},
        description="查询所有用户接口",
    )
    def list(self, request, *args, **kwargs):
        return super().list(request, *args, **kwargs)

    def update(self, request, *args, **kwargs):
        return super().update(request, *args, **kwargs)


class UserCreateSets(GenericViewSet, CreateModelMixin):
    queryset = Users.objects
    serializer_class = UserModelSerializer
    lookup_field = "username"

    @extend_schema(
        request=UserModelSerializer,
        responses={status.HTTP_201_CREATED: UserModelSerializer},
        description="用户注册接口",
        examples=[
            OpenApiExample(
                "Example 1", value={"username": "DSGLAB", "password": "FUDAN"}
            )
        ],
    )
    def create(self, request, *args, **kwargs):
        return super().create(request, *args, **kwargs)
