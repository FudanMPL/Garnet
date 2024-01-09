<script setup>
import { userRegistService, userLoginService } from '@/api/user.js'
import { watch, ref} from 'vue'
import { useUserStore } from '../../stores';
// import { userRegisterService } from '@/api/user.js'
import { useRouter } from 'vue-router'
const isRegister = ref(false)
const formModel = ref({
  username: '',
  password: '',
  repassword: ''
})
const rules = {
  username: [
    { required: true, message: '请输入用户名', trigger: 'change' },
    {
      min: 6,
      max: 12,
      message: '用户名的长度应在6-12个字符之间',
      trigger: 'blur'
    }
  ],
  password: [
    { required: true, message: '请输入密码', trigger: 'change' },
    {
      pattern: /^\S{6,15}$/,
      message: '密码为6-15位的非空字符',
      trigger: 'change'
    }
  ],
  repassword: [
    { required: true, message: '请再次输入密码', trigger: 'change' },
    {
      pattern: /^\S{6,15}$/,
      message: '密码为6-15位的非空字符',
      trigger: 'blur'
    },
    {
      validator: (rule, value, callback) => {
        if (value !== formModel.value.password) {
          callback(new Error('两次输入的密码需一致'))
        } else {
          callback()
        }
      },
      trigger: 'blur'
    }
  ]
}
const form = ref()
const register = async () => {
  // 对表单进行校验
  await form.value.validate()
  try {
  const response = await userRegistService(formModel.value);
  console.log(response)
  if (response.status == 201) {
    // 注册成功
    ElMessage.success('注册成功');
    isRegister.value = false;
  } else {
    // 注册失败，根据状态码或其他信息来处理
    ElMessage.error('注册失败，该账号已被注册');
    // 其他处理注册失败的逻辑
    // 清空表单
    cleanformModel()
  }
  } catch (error) {
  // 捕获到异常，可以处理异常情况
  ElMessage.error('注册失败，该账号已被注册');
  // 清空表单
  cleanformModel()
  }
}
const router = useRouter()
const userStore = useUserStore()

const login = async () => {
 
    await form.value.validate()
    try{
    const res  = await userLoginService(formModel.value);
    if(res.status == 200){
    console.log(res)
    userStore.setToken(res.data.token)
    ElMessage.success('登录成功')
    router.push('/locate/locateCompute')}
    else {
    // 注册失败，根据状态码或其他信息来处理
    ElMessage.error('登录失败，账号名或密码错误');
    // 其他处理注册失败的逻辑
    // 清空表单
    cleanformModel()
  }
} catch (error) {
  // 捕获到异常，可以处理异常情况
  ElMessage.error('登录失败，账号名或密码错误');
  // 其他处理异常的逻辑
  // 清空表单
  cleanformModel()
}
}

const cleanformModel = () => {
  formModel.value = {
    username: '',
    password: '',
    repassword: ''
  }
}
watch(isRegister, () => {
  cleanformModel()
})
</script>

<template>
  <el-row class="login-page">
    <el-col :span="12" class="bg">
      <span class="additional-text">Garnet</span>
      <span class="title">安全多方学习框架</span>
    </el-col>
    <el-col :span="6" :offset="3" class="form">
      <el-form
        :model="formModel"
        :rules="rules"
        ref="form"
        size="large"
        autocomplete="off"
        v-if="isRegister"
      >
        <el-form-item>
          <h1>注册</h1>
        </el-form-item>
        <el-form-item prop="username">
          <el-input
            v-model="formModel.username"
            :prefix-icon="User"
            placeholder="请输入用户名"
          ></el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="formModel.password"
            :prefix-icon="Lock"
            type="password"
            placeholder="请输入密码"
          ></el-input>
        </el-form-item>
        <el-form-item prop="repassword">
          <el-input
            v-model="formModel.repassword"
            :prefix-icon="Lock"
            type="password"
            placeholder="请再次输入密码"
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-button
            @click="register"
            class="button"
            type="primary"
            auto-insert-space
          >
            注册
          </el-button>
        </el-form-item>
        <el-form-item class="flex">
          <el-link type="info" :underline="false" @click="isRegister = false">
            ← 返回
          </el-link>
        </el-form-item>
      </el-form>
      <el-form
        :model="formModel"
        :rules="rules"
        ref="form"
        size="large"
        autocomplete="off"
        v-else
      >
        <el-form-item>
          <h1>登录</h1>
        </el-form-item>
        <el-form-item prop="username">
          <el-input
            v-model="formModel.username"
            :prefix-icon="User"
            placeholder="请输入用户名"
          ></el-input>
        </el-form-item>
        <el-form-item prop="password">
          <el-input
            v-model="formModel.password"
            name="password"
            :prefix-icon="Lock"
            type="password"
            placeholder="请输入密码"
          ></el-input>
        </el-form-item>
        <el-form-item class="flex">
          <div class="flex"></div>
        </el-form-item>
        <el-form-item>
          <el-button
            @click="login"
            class="button"
            type="primary"
            auto-insert-space
            >登录</el-button
          >
        </el-form-item>
        <el-form-item class="flex">
          <el-link type="info" :underline="false" @click="isRegister = true">
            注册 →
          </el-link>
        </el-form-item>
      </el-form>
    </el-col>
  </el-row>
</template>

<style lang="scss" scoped>
.login-page {
  height: 100vh;
  background-color: #fff;
  .bg {
    background:
      url('@/assets/logo2.png') no-repeat 5% 5% / 50px auto,
      url('@/assets/login_bg.jpg') no-repeat left / cover;
    border-radius: 0 20px 20px 0;
  }
  .form {
    display: flex;
    flex-direction: column;
    justify-content: center;
    user-select: none;
    .title {
      margin: 0 auto;
    }
    .button {
      width: 100%;
    }
    .flex {
      width: 100%;
      display: flex;
      justify-content: space-between;
    }
  }
}
.additional-text {
  position: absolute; /* 相对于父元素定位 */
  top: 7.5%; /* 居中垂直位置 */
  left: 10%; /* 调整水平位置，与第一张图片相邻 */
  transform: translate(-50%, -50%); /* 居中定位的技巧 */
  color: #fff; /* 文字颜色，根据实际情况设置 */
  text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.5);
  font-size: 50px;
  font-family: 'Times New Roman', Times, serif;
}

.title {
  position: absolute; /* 相对于父元素定位 */
  top: 20%; /* 居中垂直位置 */
  left: 7%;
  text-align: center;
  transform: translate(-0%, -50%); /* 居中定位的技巧 */
  color: #fff; /* 文字颜色，根据实际情况设置 */
  font-size: 80px;
  font-family: 'SimSun', sans-serif;
  text-shadow: 2px 2px 20px rgba(0, 0, 0, 0.5); /* 水平偏移、垂直偏移、模糊半径和颜色 */
}
</style>
