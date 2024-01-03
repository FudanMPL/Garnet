<template>
  <el-col :span="16" :offset="4">
    <div class="text-container">
      <p class="centered-text">已连接的服务器</p>
    </div>
    <!-- <br /> -->
    <div class="right-align">
      <el-button @click="fresh">刷新</el-button>
      <el-button @click="centerDialogVisible = true">
        与其他服务器建立连接
      </el-button>
    </div>
    <el-table :data="tableData" style="width: 100%">
      <el-table-column
        prop="servername"
        label="服务器名称"
        style="width: 50%"
      />
      <el-table-column prop="ip" label="IP地址" style="width: 25%" />
      <el-table-column prop="port" label="端口号" style="width: 25%" />
    </el-table>
    <el-dialog v-model="centerDialogVisible" title="Warning" width="30%" center>
      <span> 请指定需要进行连接的服务器IP和端口 </span>
      <el-form :model="connectInfo" label-width="120px">
        <el-form-item label="IP地址">
          <el-input v-model="connectInfo.ip" />
        </el-form-item>
        <el-form-item label="端口号">
          <el-input v-model="connectInfo.port" />
        </el-form-item>
      </el-form>
      <template #footer>
        <span class="dialog-footer">
          <el-button type="primary" @click="sendLink"> 确认连接 </el-button>
        </span>
      </template>
    </el-dialog>
    <el-dialog
      v-model="feedbackDialogVisible"
      title="提示"
      width="30%"
      :before-close="feedbackDialogClose"
    >
      <span class="formatted-text">{{ feedbackMessage }}</span>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="feedbackDialogVisible = false">好的</el-button>
        </span>
      </template>
    </el-dialog>
  </el-col>
</template>

<script lang="ts" setup>
import { ref, onMounted } from 'vue'
import { userLinkServer, getConnectedServer } from '../../api/user.js'

onMounted(() => {
  // 在组件被挂载到 DOM 后调用的函数
  fresh()
})

//反馈窗口设置
const feedbackDialogClose = () => {
  feedbackDialogVisible.value = false
}
const feedbackDialogVisible = ref(false)
const feedbackMessage = ref()

const sendLink = async () => {
  await userLinkServer(connectInfo.value)
  centerDialogVisible.value = false
  feedbackMessage.value='连接成功，可以加入该服务器上的任务了！'
  feedbackDialogVisible.value = true
  fresh()//连接后刷新页面
}
const centerDialogVisible = ref(false)
const connectInfo = ref({
  ip: '',
  port: ''
})

const tableData = ref([])

const taskTableLoading = ref(false)
const fresh = async () => {
  taskTableLoading.value = true
  const res = await getConnectedServer()
  tableData.value = res.data
  taskTableLoading.value = false
}
</script>

<style>
.right-align {
  display: flex;
  justify-content: flex-end;
}
.text-container {
  text-align: center;
}

.centered-text {
  font-size: 40px;
}
</style>
