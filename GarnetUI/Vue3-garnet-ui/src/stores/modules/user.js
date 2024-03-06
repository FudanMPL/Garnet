import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useUserStore = defineStore(
  'big-user',
  () => {
    const token = ref('')
    const isLogin = ref(false)
    const setToken = (newToken) => {
      token.value = newToken
    }
    const removeToken = () => {
      token.value = ''
    }
    return {
      token,
      setToken,
      removeToken,
      isLogin
    }
  },
  {
    persist: true
  }
)

export const getBaseURL = defineStore(
  'shared',
  () => {
    const BaseURL = ref('http://10.176.34.171:8000/api/')
    const updateBaseURL = (newValue) => {
      BaseURL.value = newValue
    }

    return {
      BaseURL,
      updateBaseURL
    }
  },
  {
    persist: true
  }
)
