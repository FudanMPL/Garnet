import { createPinia } from 'pinia'
import persist from 'pinia-plugin-persistedstate'

const pinia = createPinia().use(persist)
pinia.use(persist)

export default pinia

import { useUserStore } from './modules/user'
export { useUserStore }
