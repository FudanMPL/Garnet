from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from .models import Users

userData = {"username": "DSGLAB", "password": "FUDAN"}
token: str = ""


class UsersTests(APITestCase):
    def setUp(self):
        url = reverse("register")
        self.client.post(url, userData, format="json")

    def test_register(self):
        url = reverse("register")
        response = self.client.post(url, userData, format="json")
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertEqual(
            Users.objects.get(username=userData["username"]).username,
            userData["username"],
        )

        newUserData = {"username": "DSGLAB22", "password": "FUDAN"}
        response = self.client.post(url, newUserData, format="json")
        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        self.assertEqual(
            Users.objects.get(username=newUserData["username"]).username,
            newUserData["username"],
        )

    def test_login(self):
        url = reverse("login")
        response = self.client.post(url, userData, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)


class AuthenticationTests(APITestCase):
    def setUp(self):
        url = reverse("register")
        self.client.post(url, userData, format="json")
        url = reverse("login")
        token: str = self.client.post(url, userData, format="json").data["token"]
        self.assertIsNotNone(token)

    def test_get_user(self):
        url = reverse("get_user")
        response = self.client.get(url, format="json")
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["username"], userData["username"])
