from rest_framework.pagination import PageNumberPagination


class PagePagination(PageNumberPagination):
    page_size = 2  # 每页显示多少条
    page_size_query_param = "size"  # URL中每页显示条数的参数
    page_query_param = "page"  # URL中页码的参数
    max_page_size = None  # 最大页码数限制
