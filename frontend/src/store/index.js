import { createPinia } from 'pinia'

// Pinia 根实例；各业务 store（如数据集、训练任务状态）由后续任务在 src/store/ 下新增
const pinia = createPinia()

export default pinia
