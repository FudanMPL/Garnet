from Model.models import Servers
from Model.serializers import ServersModelSerializer
from rest_framework.response import Response
from rest_framework.viewsets import GenericViewSet
from rest_framework import status


class ServersListView(GenericViewSet):
    def list(self, request):
        servers = Servers.objects.all().exclude(id=1)
        return Response(
            status=status.HTTP_200_OK,
            data=ServersModelSerializer(instance=servers, many=True).data,
        )

    def listAll(self, request):
        servers = Servers.objects.all()
        return Response(
            status=status.HTTP_200_OK,
            data=ServersModelSerializer(instance=servers, many=True).data,
        )

    def get_queryset(self):
        return None
