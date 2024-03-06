<template>
  <el-col :span="18" :offset="3">
  <el-form  :model="form" :rules="rules" label-width="200px" ref="data1">
    <el-form-item prop = "taskName" label="任务名">
      <el-input v-model="form.taskName" class="inputlength" />
    </el-form-item>
  </el-form>
  <el-form inline="true" :model="form" :rules="rules" label-width="200px" ref="data2">
    <el-form-item label="模型" prop="mpc">
      <el-select v-model="form.mpc" placeholder="请选择模型">
        <el-option v-for="(item, index) in modelPartyArray"
          :key="index"
          :label="item.label"
          :value="item.value" />
      </el-select>
    </el-form-item>
  </el-form>
  <el-form inline="true" :model="form" :rules="rules" label-width="200px" ref="data4">
    <el-form-item prop = 'pN' label="运算方数量">
      <el-input v-model="form.pN" />
    </el-form-item>
  </el-form>
  <el-form inline = "true" label-width="200px">
    <el-form-item label="高级选项" >
    <el-switch v-model="moreChoice"  />
    </el-form-item>
  </el-form>
  <el-form v-show="moreChoice" inline = "true" :model="form" :rules="rules" label-width="200px" >
    
    <el-form-item label="协议" prop="protocol">
      <el-select v-model="form.protocol" placeholder="请选择使用的协议">
        <el-option v-for="(item, index) in partyArray"
          :key="index"
          :label="item.label"
          :value="item.value" />
      </el-select>
    </el-form-item>
    <el-form-item label="运行参数">
      <el-input v-model="form.protocol_parameters" />
    </el-form-item>
  </el-form>
  <el-form v-show="moreChoice" inline="true" :model="form" :rules="rules" label-width="200px" >
    <el-form-item label="编译参数">
      <el-input v-model="form.mpc_parameters" />
    </el-form-item>
  </el-form>
  <el-form :model="form" label-width="200px">
    <el-form-item label="描述">
      <el-input v-model="form.description" type="textarea" class="inputlength" />
    </el-form-item>
    <el-form-item>
    <el-dialog
    v-model="dialogVisible"
    title="请选择你的文件"
    width="60%"
    :before-close="handleClose"
  >
    <div v-for="index in pNnum" :key="index">
      <el-button  type="primary" @click="openFileTable(index-1)">选择运算方{{ index }}的文件</el-button>
    </div>
    <el-dialog
        v-model="innerVisible"
        width="60%"
        title="Inner Dialog"
        append-to-body
      >
    <el-table 
    :data="tableData" 
    style="width: 100%" 
    highlight-current-row 
    @current-change="handleCurrentChange">
        <el-table-column prop="id" label="文件id" width="180" />
        <el-table-column prop="fileName" label="文件名" width="180" />
        <el-table-column prop="file" label="文件下载链接" width="180" />
        <el-table-column prop="create_time" label="创建时间" width="180" />
        <el-table-column prop="user" label="用户" />
        <el-table-column type="selection" width="55" />
      </el-table>
      <span class="dialog-footer">
        <el-button type="primary" @click="addFile">
          确认
        </el-button>
      </span>
    </el-dialog>
    <input type="file" ref="fileInput" @change="handleFileChange" />
    <el-button @click="uploadFile">上传文件</el-button>
    <template #footer>
      <span class="dialog-footer">
        <el-button @click="freshFiles">刷新</el-button>
        <el-button @click="dialogVisible = false">返回</el-button>
        <el-button type="primary" @click="fileTotask">
          确认
        </el-button>
      </span>
    </template>
  </el-dialog>
      <el-button type="primary" @click="submitTask">提交任务</el-button>
    </el-form-item>
  </el-form>
  <el-dialog
    v-model="submitTaskSuccessDialogVisible"
    title="Tips"
    width="30%"
    :before-close="handleClose"
  >
    <span>您已经成功创建了一个任务,你现在可以在任务表中查看任务的状态</span>
    <template #footer>
      <span class="dialog-footer">
        <el-button type="primary" @click="cleanData">
          好的
        </el-button>
      </span>
    </template>
  </el-dialog>
  </el-col>
</template>

<script lang="ts" setup>
import { ref ,onMounted} from 'vue'
import { getAllModel,getAllProtocol,userCreateLocateTask,userPostData,userFileToTask } from '../../api/locateCompute.js'
import { getAllFiles } from '../../api/locateCompute.js'
// import { TaskDataItem } from '../../api/TaskDaraItem';

const moreChoice = ref(false);

const rules = {
  taskName:[{required:true,message:'请输入任务名',trigger:'change'}],
  pN: [
    { required: true, message: '请输入参与方数量', trigger: 'change' },
    {
      pattern: /^[0-9]+$/,
      message: '只能输入数字',
      trigger: 'change'
    }
  ],
  mpc:[{required:true,message:'请选择模型',trigger:'change'}],
  protocol:[{required:true,message:'请选择协议',trigger:'change'}],
}
const handleCurrentChange = (val) => {
  if(val!=null){
  taskData1.value.data = val.id}
  console.log(taskData1.value)
}

const dialogVisible = ref(false);
const innerVisible = ref(false)
const tableData = ref([])
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



const protocolArray = ref([{
  id:'',
  name:''
}])
const partyArray:any = ref([]);
const modelPartyArray:any = ref([]);

const modelArray = ref([{   
  id:'',
  name:''
}])


onMounted(async() => {
      
  const res = await getAllProtocol()
  protocolArray.value=res.data.results
  partyArray.value=protocolArray.value.map(item => ({ label: item.name, value: item.id }));

  const modelRes = await getAllModel()
  modelArray.value=modelRes.data.results
  modelPartyArray.value=modelArray.value.map
  (item => ({ label: item.name, value: item.id }));
});

const taskData = ref<TaskDataItem[]>([]);
const addFile = ()=>{
  innerVisible.value = false
  taskData.value.push(Object.assign({}, taskData1.value));
}
const pNnum = ref(0)


const freshFiles =() => {
  //将运算方数量转化为数字
  pNnum.value = parseInt(form.value.pN, 10);
  dialogVisible.value = true
}
const openFileTable = async(num) => {
  taskData1.value.index = num
  innerVisible.value = true
  const res = await getAllFiles()
  tableData.value = res.data
}

const fileTotask = async() => {
  console.log(taskData.value)
  await userFileToTask(taskData.value)
  dialogVisible.value=false
}

const handleClose = (done: () => void) => {
  dialogVisible.value = false
}
const form = ref({
  taskName:'',
  mpc_parameters:'-R 64',
  protocol_parameters:'',
  pN:'',
  status:'run_up',
  description:'',
  userid:'1',
  mpc:'',
  protocol:''
})
const filePost = ref({
  content:'123',
  description: 'asa',
  userID: 1,
  fileName:'test2'
})
const data1 = ref()
const data2 = ref()
// const data3 = ref()
const data4 = ref()

const submitTaskSuccessDialogVisible = ref(false)

const cleanData =()=>{
  //将数据面板初始化
  form.value = ({
  taskName:'',
  mpc_parameters:'-R 64',
  protocol_parameters:'',
  pN:'',
  status:'run_up',
  description:'',
  userid:'1',
  mpc:'',
  protocol:''
})
//关闭弹窗
  submitTaskSuccessDialogVisible.value = false
}
const submitTask = async () => {
  //进行校验
  if(form.value.pN =='2'){
    form.value.protocol = '3'
    
  }else if (form.value.pN == '3'){
    form.value.protocol = '4'
  }else{
    console.log('参与方数量有误')
  }
  await data1.value.validate()
  await data2.value.validate()
  // await data3.value.validate()
  await data4.value.validate()
  //提交任务表
  const res = await userCreateLocateTask(form.value)
  taskData1.value.task = res.data.id
  submitTaskSuccessDialogVisible.value = true
}

const selectedFile = ref(null);

const handleFileChange = (event) => {
      // 获取用户选择的文件
      selectedFile.value = event.target.files[0];
    };

const uploadFile = async () => {
      if (!selectedFile.value) {
        alert('Please select a file');
        return;
      }

      // 读取文件内容
      const fileContent = await readFileContent(selectedFile.value)as string;
      filePost.value.content = fileContent;
      // 打印文件内容字符串
      console.log('File Content:', fileContent);
      await userPostData(filePost.value);
      // 在这里，你可以将文件内容发送到后端服务器
      // 使用你选择的方式，比如使用 axios 发送 POST 请求

      // 清空选择的文件
      selectedFile.value = null;
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

</script>

<style scoped>
.inputlength {
  width: 700px;
}
.upload-demo{
  margin:0% 200px
}
.centered-container {
  display: flex;
  justify-content: center; /* 水平居中 */
  /* align-items: center; 垂直居中 */
  min-height: 100vh; /* 确保垂直居中在整个视口高度内 */
}

.dialog-footer button:first-child {
  margin-right: 10px;
}


</style>
