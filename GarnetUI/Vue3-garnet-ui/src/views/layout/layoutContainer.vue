<script setup>
import {
  Management,
  UserFilled,
  User,
  EditPen,
  SwitchButton,
  CaretBottom
} from '@element-plus/icons-vue' //crop
import avatar from '@/assets/default.png'
</script>

<template>
  <el-container class="layout-container">
    <el-aside width="300px">
      <div class="el-aside__logo"></div>
      <el-menu
        active-text-color="#ffd04b"
        background-color="#232323"
        :default-active="$route.path"
        text-color="#fff"
        router
      >
        <!-- <el-menu-item index="/locate" class="custom-menu-item">
          <el-icon><Management /></el-icon>
          <span class="centered-text">本地计算</span>
        </el-menu-item> -->
        <el-sub-menu index="/locate">
          <template #title>
            <el-icon><Management /></el-icon>
            <span class="centered-text">单机模拟</span>
          </template>
          <el-menu-item index="/locate/locateCompute">
            <el-icon><User /></el-icon>
            <span class="centered-text">创建任务</span>
          </el-menu-item>
          <el-menu-item index="/locate/locateTaskTable">
            <el-icon><EditPen /></el-icon>
            <span class="centered-text">查看任务</span>
          </el-menu-item>
        </el-sub-menu>
        <!-- <el-menu-item index="/mpc">
          <el-icon><Promotion /></el-icon>
          <span>多方计算</span>
        </el-menu-item> -->
        <el-sub-menu index="/mpc">
          <template #title>
            <el-icon><UserFilled /></el-icon>
            <span class="centered-text">多方计算</span>
          </template>
          <el-menu-item index="/mpc/mpC">
            <el-icon><User /></el-icon>
            <span class="centered-text">创建任务</span>
          </el-menu-item>
          <el-menu-item index="/mpc/taskTable">
            <el-icon><EditPen /></el-icon>
            <span class="centered-text">我的任务</span>
          </el-menu-item>
          <el-menu-item index="/mpc/jionTask">
            <el-icon><EditPen /></el-icon>
            <span class="centered-text">加入任务</span>
          </el-menu-item>
        </el-sub-menu>
        <el-sub-menu index="/user">
          <template #title>
            <el-icon><UserFilled /></el-icon>
            <span class="centered-text">个人中心</span>
          </template>
          <el-menu-item index="/user/userData">
            <el-icon><User /></el-icon>
            <span class="centered-text">服务器设置</span>
          </el-menu-item>
          <el-menu-item index="/user/mpcCode">
            <el-icon><EditPen /></el-icon>
            <span class="centered-text">编写MPC代码</span>
          </el-menu-item>
        </el-sub-menu>
        <!-- <el-sub-menu index="/user">
          <template #title>
            <el-icon><UserFilled /></el-icon>
            <span>个人中心</span>
          </template>
          <el-menu-item index="/user/userData">
            <el-icon><User /></el-icon>
            <span>基本资料</span>
          </el-menu-item>
          <el-menu-item index="/user/userPassword">
            <el-icon><EditPen /></el-icon>
            <span>重置密码</span>
          </el-menu-item>
        </el-sub-menu> -->
      </el-menu>
    </el-aside>
    <el-container>
      <el-header>
        <div class="header-text">安全多方学习框架<strong>Garnet</strong></div>
        <!-- <el-dropdown placement="bottom-end">
          <span class="el-dropdown__box">
            <el-avatar :src="avatar" />
            <el-icon><CaretBottom /></el-icon>
          </span>
          <template #dropdown>
            <el-dropdown-menu>
              <el-dropdown-item command="profile" :icon="User"
                >基本资料</el-dropdown-item
              >
              <el-dropdown-item command="avatar" :icon="Crop"
                >更换头像</el-dropdown-item>
              
              <el-dropdown-item command="password" :icon="EditPen"
                >重置密码</el-dropdown-item
              >
              <el-dropdown-item command="logout" :icon="SwitchButton"
                >退出登录</el-dropdown-item
              >
            </el-dropdown-menu>
          </template>
        </el-dropdown> -->
      </el-header>
      <el-main>
        <router-view></router-view>
      </el-main>
      <el-footer>Garnet</el-footer>
    </el-container>
  </el-container>
</template>

<style lang="scss" scoped>
.layout-container {
  height: 100vh;

  .el-aside {
    background-color: #232323;
    &__logo {
      height: 150px;
      background: url('@/assets/logo2.png') no-repeat center / 120px auto;
    }
    .el-menu {
      justify-content: center;
      // border-right:;
    }
  }
  .el-header {
    background-color: #fff;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 150px;
    .el-dropdown__box {
      display: flex;
      align-items: center;
      .el-icon {
        color: #999;
        margin-left: 10px;
      }

      &:active,
      &:focus {
        outline: none;
      }
    }
  }
  .el-footer {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    color: #666;
  }
}
.centered-container {
  display: flex;
  justify-content: center; /* 水平居中 */
  // align-items: center; /* 垂直居中 */
  // height: 100vh; /* 可以根据需要设置容器的高度 */
}
.centered-text {
  text-align: center; /* 文本水平居中 */
  // display: inline-block; /* 使文字垂直居中生效 */
  font-size: 30px; /* 调整字体大小 */
  background-color: #232323; /* 可选：添加背景颜色以突出显示文字位置 */
  font-family: 'SimSun', sans-serif; /* 设置为微软雅黑字体 */
}
.header-text {
  text-align: center; /* 文本水平居中 */
  // display: inline-block; /* 使文字垂直居中生效 */
  font-size: 50px; /* 调整字体大小 */
  background-color: #fff; /* 可选：添加背景颜色以突出显示文字位置 */
  font-family: 'SimSun', sans-serif; /* 设置为微软雅黑字体 */
}
.custom-menu-item {
  height: 200; /* 设置高度为 40px */
}
</style>
