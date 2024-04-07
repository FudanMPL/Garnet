<template>
  <el-col :span="18" :offset="3">
    <el-button @click="fresh">刷新</el-button>
    <el-table 
    :data="tableData" 
    style="width: 100%" 
    @row-click="handleRowClick"
    v-loading="taskTableLoading"
    max-height="400"
    >
      <el-table-column label="所属服务器" prop="host" width="180">
      </el-table-column>
      <el-table-column label="服务器名" prop="servername" width="180">
      </el-table-column>
      <el-table-column label="任务id" prop="id" width="180"> </el-table-column>
      <el-table-column label="任务名" prop="taskName" width="180">
      </el-table-column
      ><el-table-column label="参与方数量" prop="pN" width="180">
      </el-table-column>
      <el-table-column label="状态" prop="status" width="180">
      </el-table-column>
      <el-table-column label="任务码" prop="prefix" width="180">
      </el-table-column>
      <el-table-column label="描述" prop="description" width="250">
      </el-table-column>
      
      <el-table-column fixed="right" label="操作" width="180" header-align="center">
        <template v-slot="scope">
          <el-button
            link
            type="primary"
            @click="jionTask(scope.row.id,scope.row.pN,scope.row.prefix,scope.row.servername,scope.row.status)"
            size="small"
            style="text-align: center"
            >参与该任务</el-button
          >
        </template>
      </el-table-column>
    </el-table>
    
  <el-dialog
    v-model="choosePartDialogVisible"
    title="请选择你的运算方
    （发起方默认为第0方）"
    width="40%"
  >
    <el-form  :model="jionTaskData"  label-width="" >
    <el-form-item  label="运算方">
      <el-select v-model="jionTaskData.part" placeholder="请选择你的运算方">
        <el-option v-for="item in choosePartyArray"
          :key="item"
          :label="'运算方'+item"
          :value="item" />
      </el-select>
    </el-form-item>
  </el-form>
    <template #footer>
      <span class="dialog-footer">
        <el-button type="primary" @click="submitTaskPart">
          确认
        </el-button>
      </span>
    </template>
  </el-dialog>
  </el-col>
  
</template>

<script lang="ts" setup>
import { userPostData,getAllFiles,userFileToTask } from '../../api/locateCompute.js'
import { getAlltask,getAllServers } from '../../api/mpC.js'
import { ref,onMounted,toRefs} from 'vue'
import request from '../../utils/request'
import { getConnectedServer } from '../../api/user.js'
import axios from 'axios';

//进入页面时刷新任务表
onMounted(() => {
 // 在组件被挂载到 DOM 后调用的函数
 fresh();
});


//下拉框选择第几方
const choosePartyArray:any = ref([]);
const setChoosePartyArray = (num)=>{
  choosePartyArray.value = Array.from({ length: num - 1 }, (_, index) => index + 1);
};

const jionTaskData = ref({
  serverID:'',
  prefix:'',
  part:'',
  total:'',
  pN:'',
})
const choosePartDialogVisible = ref(false)
//加入任务的逻辑
const apiJoinUrl = ref('/task/remote/jion/')
const testJionUrl = ref('/task/remote/jion/')
const jionTask = async(id,pN,prefix,servername,status)=>{
  if(status =='等待参与方加入'){
    //打开弹窗 并传输数据
    choosePartDialogVisible.value = true
    jionTaskData.value.prefix = prefix
    jionTaskData.value.pN = pN
    setChoosePartyArray(pN)
    //想办法获得ID
    //先获得所有服务器数据
    const serveData = await getAllServers()
    const result = serveData.data.find(item => item.servername == servername);
    console.log(result)
    jionTaskData.value.serverID = result.id
    //根据servename查找 并返回ID
  }else{
    feedbackMessage.value = '当前无法加入该任务！'
    feedbackDialogVisible.value = true
  }
}

const tableData = ref([])
const submitTaskPart = async()=>{
  jionTaskData.value.total = '/task/remote/join/'+jionTaskData.value.serverID + '/' + jionTaskData.value.prefix + '/' + jionTaskData.value.part;
  await request.get(jionTaskData.value.total)
  choosePartDialogVisible.value = false
  feedbackMessage.value='您已经成功加入任务，可以在“我的任务”中查看并进行任务了！'
  fresh()
  feedbackDialogVisible.value = true
}

//关闭上传文件对话框
const fileDialogHandleClose = () => {
  fileDialogVisible.value = false
}

const feedbackDialogClose = () => {
  feedbackDialogVisible.value = false
}



const handleRowClick = (row) => {
  // 如果你希望点击整行时触发其他操作，可以在这里添加相应的逻辑
}

const pNnum = ref(0)//存储运算方
//变量定义 分别是选择运算方和指定文件的窗口的变量
const fileDialogVisible = ref(false)
const innerVisible = ref(false)




async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const feedbackDialogVisible = ref(false)
const feedbackMessage = ref()
const nowTaskID = ref()
const nowFileID = ref()
const fileTotask = async() => {
  const fileToTaskUrl = 'task/remote/data/'
  await request.post(fileToTaskUrl+nowTaskID.value,{"data":nowFileID.value})
  fileDialogVisible.value=false
  feedbackMessage.value='指定数据成功,稍后可以运行了！'
  feedbackDialogVisible.value = true
  await sleep(1000);
  fresh()
}

const requestArray = ref([]);
//刷新的逻辑
const taskTableLoading = ref(false)
const fresh = async () => {
  tableData.value = ([])
  taskTableLoading.value = true
  const res = await getConnectedServer()//获取所有的信息 除了自己
  // console.log(res.data)
  requestArray.value = res.data.map(item => `http://${item.ip}:${item.port}/`);
  // console.log(requestArray.value)
  // tableData.value = res.data
  // const res = await getAlltask()
  const taskData = ref()
  requestArray.value.forEach(async item => {
        // 处理每个元素的逻辑
        const getTaskURL = 'api/task/remote/model/release/'
        const realurl = ref()
        realurl.value = item+getTaskURL
        const res = await axios.get(realurl.value)
        tableData.value = tableData.value.concat(res.data);
      });
  taskTableLoading.value = false
}




</script>

<style>
.formatted-text {
  white-space: pre-line;
}
.table-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100%;
}
</style>
