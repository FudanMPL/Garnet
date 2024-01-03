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
      <el-table-column label="任务id" prop="id" width="180"> </el-table-column>
      <el-table-column label="任务名" prop="taskName" width="180">
      </el-table-column
      ><el-table-column label="参与方数量" prop="pN" width="180">
      </el-table-column>
  
      <el-table-column label="任务码" prop="prefix" width="180">
      </el-table-column>
      <el-table-column label="描述" prop="description" width="250">
      </el-table-column>
      <el-table-column label="状态" prop="status" width="180">
      </el-table-column>
      <el-table-column fixed="right" label="操作" width="180" header-align="center">
        <template v-slot="scope">
          <el-button
            link
            type="primary"
            @click="jionTask(scope.row.id,scope.row.pN,scope.row.prefix,scope.row.host,scope.row.status)"
            size="small"
            >参与该任务</el-button
          >
        </template>
      </el-table-column>
    </el-table>
    <el-dialog
      v-model="resultDialogVisible"
      title="执行结果："
      width="60%"
      :before-close="resultDialogClose"
    >
    <el-col :span="20" :offset="2">
      <el-table :data="result1Data" style="width: 100%" max-height="400">
      <el-table-column v-for="i in column" :key="i" :prop="i" label='' width="180" />
      </el-table>
    </el-col>
      <template #footer>
        <span class="dialog-footer">
          <el-button @click="resultDialogVisible = false">好的</el-button>
          <el-button type="primary" @click="downloadResultFile" :disabled="downloadDisabled">
            下载
          </el-button>
          <a ref="downloadLink" style="display: none"></a>
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
    <el-dialog
    v-model="fileDialogVisible"
    title="请选择你的文件"
    width="60%"
    :before-close="fileDialogHandleClose"
  >
    <!-- <div v-for="index in pNnum" :key="index">
      <el-button  type="primary" @click="openFileTable(index-1)">选择运算方{{ index }}的文件</el-button>
    </div> -->
    <el-table 
    :data="fileTableData" 
    style="width: 100%" 
    highlight-current-row 
    @current-change="handleCurrentChange">
        <el-table-column prop="id" label="文件id" width="180" />
        <el-table-column prop="fileName" label="文件名" width="180" />
        <el-table-column prop="create_time" label="创建时间" width="180" />
        <el-table-column prop="user" label="用户" />
      </el-table>
    <!-- <span class="dialog-footer">
        <el-button type="primary" @click="addFile">
          确认
        </el-button>
    </span> -->
    <!-- <span>提示：如果服务器中没有你的文件，你可以在下方上传你的文件后再进行指定</span> -->
    <br>
    <el-text class="mx-1" type="primary">提示：</el-text>
    <el-text class="mx-1">如果服务器中没有你的文件，你可以在下方上传你的文件后再进行指定</el-text>
    <br>
    <br>
    <!-- <el-dialog
        v-model="innerVisible"
        width="60%"
        title="请选择你的文件"
        append-to-body
      >
    <el-table 
    :data="fileTableData" 
    style="width: 100%" 
    highlight-current-row 
    @current-change="handleCurrentChange">
        <el-table-column prop="id" label="文件id" width="180" />
        <el-table-column prop="fileName" label="文件名" width="180" />
        <el-table-column prop="create_time" label="创建时间" width="180" />
        <el-table-column prop="user" label="用户" />
    </el-table>
      <span class="dialog-footer">
        <el-button type="primary" @click="addFile">
          确认1
        </el-button>
      </span>
    </el-dialog> -->
    <input type="file" ref="fileInput" @change="handleFileChange" />
    <el-button @click="uploadFile">上传文件</el-button>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="fileDialogVisible = false">返回</el-button>
        <el-button type="primary" @click="fileTotask">
          确认
        </el-button>
      </span>
    </template>
  </el-dialog>
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
const jionTask = async(id,pN,prefix,host,status)=>{
  if(status =='等待参与方加入'){
    //打开弹窗 并传输数据
    choosePartDialogVisible.value = true
    jionTaskData.value.prefix = prefix
    jionTaskData.value.pN = pN
    setChoosePartyArray(pN)
    //想办法获得ID
    //先获得所有服务器数据
    const serveData = await getAllServers()
    const result = serveData.data.find(item => item.ip == host);
    jionTaskData.value.serverID = result.id
    //根据host查找 返回ID
  }else{
    feedbackMessage.value = '当前无法加入该任务！'
    feedbackDialogVisible.value = true
  }
}
const submitTaskPart = async()=>{
  jionTaskData.value.total = '/task/remote/join/'+jionTaskData.value.serverID + '/' + jionTaskData.value.prefix + '/' + jionTaskData.value.part;
  console.log(jionTaskData.value.total)
  await request.get(jionTaskData.value.total)
  choosePartDialogVisible.value = false
  feedbackMessage.value='您已经成功加入任务，可以在“我的任务”中查看并进行任务了！'
  feedbackDialogVisible.value = true
}



//删除任务的逻辑
const apiDeleteUrl = ref('/task/remote/model/')
const testDeleteUrl = ref('/task/remote/model/')
const deleteTaskByID = async(id) =>{
  testDeleteUrl.value = apiDeleteUrl.value + id
  console.log(testDeleteUrl.value)
  await request.delete(testDeleteUrl.value)
  fresh()
  feedbackMessage.value= '任务'+id+'已被删除！'
  feedbackDialogVisible.value = true
}

//数据结构定义 任务表
const tableData = ref([])
// 定义数据结构，上传的文件
const filePost = ref({
  content:'',
  description: '',
  userID: 1,
  fileName:''
})


//关闭上传文件对话框
const fileDialogHandleClose = () => {
  fileDialogVisible.value = false
}

const feedbackDialogClose = () => {
  feedbackDialogVisible.value = false
}

//运行任务的逻辑
const apiRunUrl = ref('/task/remote/run/')
const testRunUrl = ref('/task/remote/run/')
const handleRun = async(id,status) => {
  if(status == '本地就绪'){
  testRunUrl.value = apiRunUrl.value + id
  await request.get(testRunUrl.value)
  feedbackMessage.value = '该任务已成功运行！'
  fresh()
  feedbackDialogVisible.value = true
}
  if(status == '等待数据'){
    feedbackMessage.value = '请先为该任务指定数据！'
    feedbackDialogVisible.value = true
  }
  if(status == '运行中'){
    feedbackMessage.value = '该任务已在运行中！'
    feedbackDialogVisible.value = true
  }
  if(status == '已完成'){
    feedbackMessage.value = '该任务已运行完毕！'
    feedbackDialogVisible.value = true
  }
}



//查看并获取结果的逻辑
const message = ref()// 定义数据结构，返回的结果
const downloadDisabled = ref(false)
const resultDialogVisible = ref(false)
const apiResultUrl = ref('/task/remote/results/')
const testResultUrl = ref('/task/remote/results/')
//结果弹窗的关闭逻辑
const resultDialogClose = () => {
  resultDialogVisible.value = false
}
//查询是否已完成的逻辑
const handleResult = async (id, status) => {
  if (status == '已完成') {
    testResultUrl.value = apiResultUrl.value + id
    const res = await request.get(testResultUrl.value)
    message.value = res.data
    result1.value = res.data
    await transformData(result1)
    console.log(result1Data.value)
    console.log(column.value)
    resultDialogVisible.value = true
    downloadDisabled.value = false
  } 
  if(status == '运行中') {
    message.value = '计算还未已完成！'
    downloadDisabled.value = true
    resultDialogVisible.value = true
  }
  if(status == '就绪'){
    feedbackMessage.value='该任务还未运行！'
    feedbackDialogVisible.value = true
  }
  if(status == '等待数据'){
    feedbackMessage.value='请先为该任务指定数据！'
    feedbackDialogVisible.value = true
  }
}



const handleRowClick = (row) => {
  // 如果你希望点击整行时触发其他操作，可以在这里添加相应的逻辑
}

const pNnum = ref(0)//存储运算方
//变量定义 分别是选择运算方和指定文件的窗口的变量
const fileDialogVisible = ref(false)
const innerVisible = ref(false)
const handleUploadFile = async(id,pN,status) =>{
  // if((status == '等待数据')||(status == '就绪'))
  if(true){
  pNnum.value = pN
  taskData1.value.task = id
  nowTaskID.value = id
  const res = await getAllFiles()
  fileTableData.value = res.data
  fileDialogVisible.value = true}
  else{
  }
}



//变量定义 指定任务数据和临时变量
const taskData1=ref({
  index:'',
  data:'',
  task:''
})
interface TaskDataItem {
  index: string;
  data: string;
  task: string;
}
const fileTableData = ref([])
const addTaskData = ref(true)
const taskData = ref<TaskDataItem[]>([]);
//将临时文件加入到最终上传的数据结构中 目前还存在替换的问题
const addFile = ()=>{
  innerVisible.value = false
  console.log(taskData.value);
  console.log(taskData1.value)
  taskData.value.forEach((item, index) => {
  if (item.index == taskData1.value.index) {
    taskData.value[index] = Object.assign({}, taskData1.value);;
    addTaskData.value = false
    console.log('覆盖')
  }
});
if(addTaskData.value == true){
  console.log('添加')
  taskData.value.push(Object.assign({}, taskData1.value));
  console.log(taskData.value);
  console.log(taskData.value.length);
}
addTaskData.value = true
}
const openFileTable = async(num) => {
  taskData1.value.index = num
  innerVisible.value = true
  const res = await getAllFiles()
  fileTableData.value = res.data
}

const handleCurrentChange = (val) => {//选中文件的逻辑
  if(val!=null){
  taskData1.value.data = val.id
  nowFileID.value = val.id
}
}


async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

const feedbackDialogVisible = ref(false)
const feedbackMessage = ref()
const nowTaskID = ref()
const nowFileID = ref()
const fileTotask = async() => {
//   if(taskData.value.length == pNnum.value){
//   await userFileToTask(taskData.value)
//   fileDialogVisible.value=false
//   feedbackMessage.value='指定数据成功,稍后可以运行了！'
//   await sleep(1000);
//   fresh()
// }else{feedbackMessage.value='还有数据未指定，请重新指定数据'}
// feedbackDialogVisible.value = true
  //判断是否选中了数据
  //开始上传数据
  const fileToTaskUrl = 'task/remote/data/'
  await request.post(fileToTaskUrl+nowTaskID.value,{"data":nowFileID.value})
  fileDialogVisible.value=false
  feedbackMessage.value='指定数据成功,稍后可以运行了！'
  feedbackDialogVisible.value = true
  await sleep(1000);
  fresh()
}


//上传文件的逻辑
const selectedFile = ref(null);
const handleFileChange = (event) => {
      // 获取用户选择的文件
      selectedFile.value = event.target.files[0];
    };
const uploadFile = async () => {
      if (!selectedFile.value) {
        alert('请先选择一个文件');
        return;
      }
      // 读取文件内容
      const fileContent = await readFileContent(selectedFile.value)as string;
      filePost.value.content = fileContent;
      // 打印文件内容字符串
      await userPostData(filePost.value);
      // 在这里，你可以将文件内容发送到后端服务器
      // 使用你选择的方式，比如使用 axios 发送 POST 请求
      feedbackMessage.value = '文件：'+filePost.value.fileName+'已成功上传到服务器，你可以将它指定到你的任务中了！'
      feedbackDialogVisible.value = true
      // 清空选择的文件
      selectedFile.value = null;
      //刷新文件列表
      await sleep(100)
      const res = await getAllFiles()
      console.log(res.data)
      fileTableData.value = res.data

    };
const readFileContent = (file) => {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          // 读取文件内容并将其作为字符串传递给 resolve
          resolve(reader.result);
        };

        reader.onerror = (error) => {
          // 如果发生错误，则将错误信息传递给 reject
          reject(error);
        };

        // 以文本形式读取文件内容
        filePost.value.fileName=file.name
        reader.readAsText(file);
      });
    };

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
        // console.log(realurl.value)
        const res = await axios.get(realurl.value)
        tableData.value = tableData.value.concat(res.data);
        // console.log(res.data)
        // tableData.value.push(res.data);
        
      });
  // console.log(res.data)
  // console.log(tableData.value)
  // tableData.value = taskData.value
  taskTableLoading.value = false
}

//结果展示
const result1Data = ref([]);
const result1 = ref();
const column = ref();

const transformData=(rowData)=>{

  const lines = rowData.value.trim().split('\n');
  const data:any= [];
  let isDataSection = false;

  for (const line of lines) {
        if (line.startsWith('========') ) {
          isDataSection = !isDataSection;
        } else if (isDataSection && line.trim() !== '') {
          const values = (line.trim() as string).split(/\s+/); // 使用类型断言确保 line 是字符串

          // const rowData = {
          //   startPrescriptionNumber: values[0],
          //   startDate: values[1],
          //   endPrescriptionNumber: values[2],
          //   endDate: values[3],
          //   patientID: values[4],
          //   exceedingMultiple: values[5]
          // };
          const rowData = {};
          for (let i = 1; i <= values.length;i++){
            rowData[String(i)] = values[i-1].replace(",", "");
          }
          data.push(rowData);
        }

      }
      column.value = Object.keys(data[0]).length; 
      result1Data.value = data;
}

const downloadLink = ref();

const downloadResultFile = async() => {
  const testURL = 'http://10.176.34.171:8000/api'+testResultUrl.value
  const response = await fetch(testURL);
  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  downloadLink.value.href = url;
  console.log( testResultUrl.value)
  console.log( downloadLink.value.herf)
  downloadLink.value.href = url;
  const currentDate = new Date();
  downloadLink.value.download = '下载数据-'+currentDate+'.txt';
  downloadLink.value.click();
  resultDialogVisible.value = false;
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
