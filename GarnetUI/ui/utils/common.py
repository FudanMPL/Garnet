import datetime
import hashlib
import os
import subprocess
from pathlib import Path
from typing import List, Tuple

import requests
from django.conf import settings


def md5(s, sale=settings.SECRET_KEY, encoding="utf-8"):
    """

    :param s: 加密内容，字节或字符串
    :param sale: 盐
    :param encoding: 编码
    :return: 返回16进制字节
    """
    if not isinstance(s, bytes):
        s = str(s).encode(encoding=encoding)
    md = hashlib.md5()
    md.update(s)
    if not sale:
        return md.hexdigest()
    else:
        if not isinstance(sale, bytes):
            sale = str(sale).encode(encoding=encoding)
        md.update(sale)
        return md.hexdigest()


def inputFileLink(filepaths: List[Tuple[int, str]], prefix: str):
    """
    输入数据软链接
    """
    for index, filepath in filepaths:
        subprocess.Popen(
            f"ln -s {settings.MEDIA_ROOT}/data/{filepath} {settings.GARNETPATH}/Player-Data/{prefix}-P{index}-0",
            shell=True,
        )


def download(url: str, path: Path):
    """
    文件下载
    """
    if not os.path.exists(path.parents[0]):
        os.makedirs(path.parents[0])
    with open(file=path, mode="wb") as f:
        f.write(requests.get(url).content)
        f.close()
