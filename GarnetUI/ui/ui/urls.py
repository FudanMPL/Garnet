"""
URL configuration for ui project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path
from drf_spectacular.views import (
    SpectacularAPIView,
    SpectacularRedocView,
    SpectacularSwaggerView,
)

from .views import (
    link,
    localTask,
    login,
    mpc,
    protocol,
    remotetask,
    servers,
    user,
    userdata,
)

urlpatterns = [
    # region Model
    # region user
    path("api/model/user/", user.UserSets.as_view({"get": "list"})),
    path(
        "api/model/user/<str:username>",
        user.UserSets.as_view({"post": "partial_update", "delete": "destroy"}),
    ),
    # endregion
    # region userdata
    path(
        "api/model/userdata/",
        userdata.UserdataSets.as_view({"get": "list", "post": "create"}),
    ),
    path(
        "api/model/userdata/string/",
        userdata.UserdataFormStringView.as_view({"post": "str2file"}),
    ),
    # endregion
    # region mpc
    path("api/model/mpc/", mpc.MpcSets.as_view({"get": "list", "post": "create"})),
    path(
        "api/model/mpc/<int:pk>",
        mpc.MpcSets.as_view(
            {"get": "retrieve", "post": "partial_update", "delete": "destroy"}
        ),
    ),
    path(
        "api/model/mpc/string/",
        mpc.MpcFormStringView.as_view({"post": "str2file"}),
    ),
    path(
        "api/model/mpc/compile/",
        mpc.MpcFormStringView.as_view({"post": "tryCompile"}),
    ),
    # endregion
    # region protocol
    path(
        "api/model/protocol/",
        protocol.ProtocolSets.as_view({"get": "list", "post": "create"}),
    ),
    # endregion
    # region servers
    path("api/model/servers/", servers.ServersListView.as_view({"get": "list"})),
    path("api/model/servers/all/", servers.ServersListView.as_view({"get": "listAll"})),
    # endregion
    # endregion
    # region Docs
    path("api/docs/schema/", SpectacularAPIView.as_view(), name="schema"),
    path(
        "api/docs/swagger-ui/",
        SpectacularSwaggerView.as_view(url_name="schema"),
        name="swagger-ui",
    ),
    path(
        "api/docs/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"
    ),
    # endregion
    # region Task
    # region Local
    path(
        "api/task/local/model/",
        localTask.LocalTaskSets.as_view({"get": "list", "post": "create"}),
    ),
    path(
        "api/task/local/model/<int:pk>",
        localTask.LocalTaskSets.as_view(
            {"get": "retrieve", "post": "partial_update", "delete": "destroy"}
        ),
    ),
    path(
        "api/task/local/data/",
        localTask.DataTaskRelationshipSets.as_view({"post": "create_update"}),
    ),
    path(
        "api/task/local/data/<int:id>",
        localTask.DataTaskRelationshipSets.as_view({"get": "retrive"}),
    ),
    path("api/task/local/results/<int:id>", localTask.LocalResultsView.as_view()),
    path("api/task/local/run/<int:taskID>", localTask.LocalRun.as_view({"get": "run"})),
    # endregion
    # region Remote
    path(
        "api/task/remote/release/",
        link.TaskReleaseView.as_view({"get": "list", "post": "release"}),
    ),
    path(
        "api/task/remote/join/<int:serverID>/<uuid:prefix>/<int:part>",
        link.TaskJoinView.as_view({"get": "join"}),
    ),
    path(
        "api/task/remote/model/",
        remotetask.RemoteTaskSets.as_view({"get": "list", "post": "create"}),
    ),
    path(
        "api/task/remote/model/release/",
        remotetask.RemoteTaskSets.as_view({"get": "release"}),
    ),
    path(
        "api/task/remote/model/<int:pk>",
        remotetask.RemoteTaskSets.as_view(
            {"get": "retrieve", "post": "partial_update", "delete": "destroy"}
        ),
    ),
    path(
        "api/task/remote/data/<int:pk>",
        remotetask.RemoteTaskAddData.as_view({"post": "partial_update"}),
    ),
    path(
        "api/task/remote/results/<int:id>",
        remotetask.RemoteResultsView.as_view({"get": "get"}),
    ),
    path(
        "api/task/remote/run/<int:taskID>/",
        remotetask.RemoteRun.as_view({"get": "coordinator"}),
    ),
    path(
        "api/task/remote/run/<uuid:prefix>/",
        remotetask.RemoteRun.as_view({"get": "participant"}),
    ),
    # endregion
    # endregion
    # region Link
    # region Sender
    path(
        "api/link/server/send/<int:taskID>",
        link.TaskReleaseView.as_view({"get": "serverSend"}),
    ),
    path("api/link/metadata/send/", link.MetadataView.as_view({"post": "send"})),
    path(
        "api/link/ready/send/<uuid:prefix>/", link.ReadyView.as_view({"get": "ready"})
    ),
    # endregion
    # region Receiver
    path("api/link/metadata/receive/", link.MetadataView.as_view({"post": "receive"})),
    path(
        "api/link/server/receive/", link.TaskJoinView.as_view({"post": "serverReceive"})
    ),
    path(
        "api/link/ready/receive/<uuid:prefix>/",
        link.ReadyView.as_view({"get": "isReady"}),
    ),
    # endregion
    # endregion
    path("admin/", admin.site.urls),
    path(
        "api/register/",
        user.UserCreateSets.as_view({"post": "create"}),
        name="register",
    ),
    path("api/login/", login.LoginView.as_view(), name="login"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
