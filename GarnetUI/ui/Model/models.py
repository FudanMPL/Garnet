import uuid
from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator


class Users(models.Model):
    """用户表"""

    username = models.CharField(
        max_length=64, unique=True, null=False, verbose_name="用户名"
    )
    password = models.CharField(max_length=255, null=False, verbose_name="密码")
    role = models.IntegerField(default=1, blank=True, null=False, verbose_name="用户权限")
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    update_time = models.DateTimeField(auto_now=True, verbose_name="更新时间")


class Servers(models.Model):
    """服务器表"""

    servername = models.CharField(
        max_length=30, unique=True, null=False, verbose_name="服务器别名"
    )
    ip = models.GenericIPAddressField(verbose_name="IP地址")
    port = models.IntegerField(verbose_name="服务器端口")
    token = models.CharField(max_length=64, null=True, blank=True, verbose_name="token")


class Mpc(models.Model):
    """mpc表"""

    name = models.CharField(max_length=50, null=False, verbose_name="mpc功能名")
    # upload_to属性用于指定上传文件的保存位置，以 settings.py 中定义的 MEDIA_ROOT 为路径前缀。如果存在同名文件，则会自动生成"_xxxxxx"的后缀
    file = models.FileField(
        upload_to="mpc",
        blank=True,
        null=True,
        verbose_name="mpc文件路径",
    )
    description = models.TextField(null=True, verbose_name="详情")


class Protocol(models.Model):
    """协议表"""

    name = models.CharField(max_length=50, null=False, verbose_name="协议名")
    description = models.TextField(null=True, verbose_name="详情")


class UserData(models.Model):
    """用户数据表"""

    user = models.ForeignKey(
        to="Users",
        to_field="id",
        on_delete=models.CASCADE,
        verbose_name="用户主键",
        blank=True,
    )
    file = models.FileField(
        upload_to="data", unique=True, blank=True, null=True, verbose_name="用户文件"
    )
    create_time = models.DateTimeField(
        auto_now_add=True, null=True, verbose_name="创建时间"
    )
    update_time = models.DateTimeField(auto_now=True, null=True, verbose_name="更新时间")
    description = models.TextField(null=True, verbose_name="详情")


class LocalTask(models.Model):
    """本地任务表"""

    taskName = models.CharField(
        max_length=40, verbose_name="任务名", unique=False, null=False
    )
    userid = models.ForeignKey(
        to="Users", to_field="id", on_delete=models.DO_NOTHING, verbose_name="创建用户"
    )
    mpc = models.ForeignKey(
        to="Mpc", on_delete=models.DO_NOTHING, verbose_name="使用的mpc文件"
    )
    mpc_parameters = models.TextField("编译参数", null=True, blank=True)
    protocol = models.ForeignKey(
        to="Protocol", on_delete=models.DO_NOTHING, verbose_name="使用的协议"
    )
    protocol_parameters = models.TextField("运行参数", null=True, blank=True)
    pN = models.PositiveIntegerField(verbose_name="运算的参与方数量")
    userdata = models.ManyToManyField(
        UserData, verbose_name="使用的数据", through="DataTaskRelationship"
    )
    prefix = models.UUIDField(default=uuid.uuid4, editable=False, verbose_name="数据前缀")
    status = models.CharField(max_length=30, verbose_name="状态", blank=True)
    create_time = models.DateTimeField(
        auto_now_add=True, null=True, verbose_name="创建时间"
    )
    run_time = models.DateTimeField(null=True, verbose_name="开始运行时间")
    end_time = models.DateTimeField(null=True, verbose_name="结束时间")
    description = models.TextField(null=True, verbose_name="详情", blank=True)


class DataTaskRelationship(models.Model):
    """用户数据表与本地任务表的关系的中间表"""

    data = models.ForeignKey(UserData, on_delete=models.CASCADE)
    task = models.ForeignKey(LocalTask, on_delete=models.CASCADE)
    index = models.IntegerField()


class RemoteTask(models.Model):
    """远端任务表"""

    taskName = models.CharField(
        max_length=40, verbose_name="任务名", unique=False, null=False
    )
    mpc = models.ForeignKey(
        to="Mpc", to_field="id", on_delete=models.DO_NOTHING, verbose_name="使用的mpc文件"
    )
    mpc_parameters = models.TextField("编译参数", null=True, blank=True)
    protocol = models.ForeignKey(
        to="Protocol", to_field="id", on_delete=models.DO_NOTHING, verbose_name="使用的协议"
    )
    protocol_parameters = models.TextField("运行参数", null=True, blank=True)
    pN = models.PositiveIntegerField(verbose_name="模拟运算的参与方数量")
    part = models.IntegerField(
        validators=[MinValueValidator(0)],
        verbose_name="本方是第几方",
    )
    servers = models.ManyToManyField(
        to=Servers, through="ServerTaskRelationship", verbose_name="参与计算的服务器"
    )
    data = models.ForeignKey(
        to=UserData,
        to_field="id",
        on_delete=models.DO_NOTHING,
        null=True,
        verbose_name="本方使用的数据",
        blank=True,
    )
    prefix = models.UUIDField(default=uuid.uuid4, editable=False, verbose_name="数据前缀")
    host = models.GenericIPAddressField(verbose_name="协调方ip地址", blank=True, null=True)
    baseport = models.IntegerField(verbose_name="基端口")
    status = models.CharField(max_length=30, verbose_name="状态", blank=True)
    create_time = models.DateTimeField(
        auto_now_add=True, null=True, verbose_name="创建时间", blank=True
    )
    run_time = models.DateTimeField(null=True, verbose_name="开始运行时间", blank=True)
    end_time = models.DateTimeField(null=True, verbose_name="结束时间", blank=True)
    description = models.TextField(null=True, verbose_name="详情", blank=True)


class ServerTaskRelationship(models.Model):
    """服务器表与远端任务表的关系的中间表"""

    server = models.ForeignKey(Servers, on_delete=models.DO_NOTHING)
    task = models.ForeignKey(RemoteTask, on_delete=models.CASCADE)
    part = models.IntegerField()


class Token(models.Model):
    """token表"""

    user = models.OneToOneField(
        to="Users", to_field="id", on_delete=models.CASCADE, verbose_name="用户主键"
    )
    token = models.CharField(max_length=64, null=True, verbose_name="token")
    expires_time = models.CharField(max_length=32, null=True, verbose_name="有效期")
    create_time = models.DateTimeField(
        auto_now_add=True, null=True, verbose_name="创建时间"
    )
    update_time = models.DateTimeField(auto_now=True, null=True, verbose_name="更新时间")
