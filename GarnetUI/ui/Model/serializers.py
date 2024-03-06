from django import forms
from rest_framework import serializers
from Model.models import *
from utils.common import md5
from django.conf import settings


class UserModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Users
        fields = ["username", "password"]

    def create(self, validated_data):
        """重写create方法实现，将密码加密后保存"""
        user = Users(**validated_data)
        user.password = md5(user.password)
        user.save()
        return user

    def update(self, instance, validated_data):
        """重写update方法实现，将密码加密后保存"""
        instance.password = md5(validated_data.get("password")) or instance.password
        instance.username = validated_data.get("username") or instance.username
        instance.save()
        return instance


class ServersModelSerializer(serializers.ModelSerializer):
    url = serializers.SerializerMethodField()

    class Meta:
        model = Servers
        fields = ["id", "servername", "ip", "url", "port", "token"]

    def get_url(self, obj) -> str:
        return f"http://{settings.IPADDRESS}:{settings.PORT}{settings.MEDIA_URL}ssl/{obj.servername}.pem"


class MpcModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mpc
        fields = "__all__"


class ProtocolModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = Protocol
        fields = "__all__"


class UserDataModelSerializer(serializers.ModelSerializer):
    fileName = serializers.SerializerMethodField()
    file = serializers.FileField(allow_empty_file=True)

    def is_valid(self, *, raise_exception=False):
        return super().is_valid(raise_exception=raise_exception)

    def get_fileName(self, obj) -> str:
        return obj.file.name.split("/")[-1]

    class Meta:
        model = UserData
        fields = [
            "id",
            "user",
            "file",
            "fileName",
            "description",
            "create_time",
            "update_time",
        ]


class LocalTaskModelSerializer(serializers.ModelSerializer):
    mpcName = serializers.SerializerMethodField()

    def get_mpcName(self, obj) -> str:
        return obj.mpc.name

    def create(self, validated_data):
        task = LocalTask(**validated_data)
        task.status = "等待数据"
        task.save()
        return task

    class Meta:
        model = LocalTask
        fields = [
            "id",
            "taskName",
            "userid",
            "mpc",
            "mpc_parameters",
            "protocol",
            "protocol_parameters",
            "pN",
            "userdata",
            "status",
            "description",
            "mpcName",
        ]


class RemoteTaskModelSerializer(serializers.ModelSerializer):
    servername = serializers.SerializerMethodField()

    def validate(self, attrs):
        if attrs["part"] > attrs["pN"] - 1:
            raise serializers.ValidationError("序号不能大于总参与方的数量")
        return super().validate(attrs)

    def create(self, validated_data):
        task = RemoteTask(**validated_data)
        if task.host == None:
            task.host = settings.IPADDRESS
        task.status = "等待参与方加入"
        task.save()
        return task

    def get_servername(self, obj) -> str:
        return Servers.objects.get(id=1).servername

    class Meta:
        model = RemoteTask
        fields = [f.name for f in model._meta.fields] + ["servername"]


class ServerTaskRelationshipModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = ServerTaskRelationship
        fields = "__all__"


class DataTaskRelationshipModelSerializer(serializers.ModelSerializer):
    dataName = serializers.SerializerMethodField()

    def get_dataName(self, obj) -> str:
        return obj.data.file.name.split("/")[-1]

    def validate(self, attrs):
        if attrs["index"] > attrs["task"].pN:
            raise serializers.ValidationError("序号大于任务数")
        return super().validate(attrs)

    def create(self, validated_data):
        try:
            relationship = DataTaskRelationship.objects.get(
                index=validated_data["index"], task=validated_data["task"]
            )
            relationship.data = validated_data["data"]
            relationship.save()
        except:
            relationship = DataTaskRelationship.objects.create(**validated_data)
        return relationship

    class Meta:
        model = DataTaskRelationship
        fields = ["data", "task", "index", "dataName"]


class MetadataSerializer(serializers.Serializer):
    ip = serializers.IPAddressField()
    port = serializers.IntegerField()


class TaskResponseSerializer(serializers.ModelSerializer):
    mpcURL = serializers.SerializerMethodField()
    protocolName = serializers.SerializerMethodField()
    mpcName = serializers.SerializerMethodField()

    def create(self, validated_data):
        return RemoteTask(**validated_data)

    class Meta:
        model = RemoteTask
        exclude = ["mpc", "protocol", "part", "servers", "data"]

    def get_mpcURL(self, obj) -> str:
        return f"http://{settings.IPADDRESS}:{settings.PORT}{settings.MEDIA_URL}{obj.mpc.file}"

    def get_mpcName(self, obj) -> str:
        return f"{obj.mpc.name}"

    def get_protocolName(self, obj) -> str:
        return f"{obj.protocol.name}"


class TaskRequestSerializer(serializers.Serializer):
    servername = serializers.CharField()
    prefix = serializers.UUIDField()
    part = serializers.IntegerField()


class JoinedServersSerializer(serializers.Serializer):
    servername = serializers.SerializerMethodField()
    serverIP = serializers.SerializerMethodField()
    serverPort = serializers.SerializerMethodField()
    part = serializers.SerializerMethodField()

    class Meta:
        model = ServerTaskRelationship
        fields = ["servername", "serverIP", "serverPort", "part"]

    def get_servername(self, obj) -> str:
        return obj.server.servername

    def get_serverIP(self, obj) -> str:
        return obj.server.ip

    def get_serverPort(self, obj) -> int:
        return obj.server.port

    def get_part(self, obj) -> int:
        return obj.part


class NoneSerializer(serializers.Serializer):
    """
    空的序列化器，只是为了使用GenericViewSet而创建的。
    """

    pass
