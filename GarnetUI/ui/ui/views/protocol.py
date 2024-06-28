from Model.models import Protocol
from Model.serializers import ProtocolModelSerializer
from rest_framework.viewsets import ModelViewSet

from ..authentication import UserAuthentication
from ..pagination import PagePagination


class ProtocolSets(ModelViewSet):
    queryset = Protocol.objects.all()
    serializer_class = ProtocolModelSerializer
    authentication_classes = [UserAuthentication]
    pagination_class = PagePagination
