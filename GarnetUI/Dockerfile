# 目前来说 封装好的镜像名为 garnet3.14 v1.0 
# 目前这一个版本只编译了两方的虚拟机
# 拉区镜像
FROM zengwen0318/garnet3.14:v1.0

# 设置环境变量
ENV HostName="garnet173"
ENV HostAddress="10.176.34.173"


# 复制启动脚本到容器中
COPY start.sh /usr/local/bin/start.sh

# 添加可执行权限
RUN chmod +x /usr/local/bin/start.sh

# 指定启动命令
CMD ["/usr/local/bin/start.sh"]
