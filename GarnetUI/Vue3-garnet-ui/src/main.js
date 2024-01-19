import { createApp } from 'vue'

import App from './App.vue'
import router from './router'
import pinia from '@/stores/index'
// import VueMeta from 'vue-meta'

const app = createApp(App)

app.use(pinia)
app.use(router)
// app.use(VueMeta)

app.mount('#app')
