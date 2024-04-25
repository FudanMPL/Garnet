import { createRouter, createWebHistory } from 'vue-router'
import { useUserStore } from '@/stores'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/login',
      component: () => import('@/views/login/loginPage.vue')
    }, //登陆页
    {
      path: '/',
      component: () => import('@/views/layout/layoutContainer.vue'),
      redirect: '/login',
      children: [
        {
          path: 'locate/locateCompute',
          component: () => import('@/views/locate/locateCompute.vue')
        },
        {
          path: 'locate/locateTaskTable',
          component: () => import('@/views/locate/locateTaskTable.vue')
        },
        {
          path: 'mpc/mpC',
          component: () => import('@/views/mpc/mpC.vue')
        },
        {
          path: 'mpc/jionTask',
          component: () => import('@/views/mpc/jionTask.vue')
        },
        {
          path: 'mpc/taskTable',
          component: () => import('@/views/mpc/taskTable.vue')
        },
        {
          path: 'user/userdata',
          component: () => import('@/views/user/userData.vue')
        },
        {
          path: 'user/mpcCode',
          component: () => import('@/views/user/mpcCode.vue')
        }
      ]
    } //布局
  ]
})


//登录访问拦截
router.beforeEach((to)=>{
  const useStore = useUserStore()
  if(!useStore.isLogin && to.path !== '/login'){
    return '/login'
  }
  return true
})


export default router
