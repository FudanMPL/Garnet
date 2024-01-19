from Model.models import Servers
from Model.serializers import ServersModelSerializer
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from rest_framework import status
from ..pagination import PagePagination


class ServersListView(GenericViewSet):
    pagination_class = PagePagination

    def list(self, request):
        servers = Servers.objects.all().exclude(id=1)
        page = self.pagination_class()
        server_page = page.paginate_queryset(
            queryset=servers, request=request, view=self
        )
        return Response(
            status=status.HTTP_200_OK,
            data=ServersModelSerializer(server_page, many=True).data,
        )

    def listAll(self, request):
        servers = Servers.objects.all()
        page = self.pagination_class()
        server_page = page.paginate_queryset(
            queryset=servers, request=request, view=self
        )
        return Response(
            status=status.HTTP_200_OK,
            data=ServersModelSerializer(server_page, many=True).data,
        )

    def get_queryset(self):
        return None
