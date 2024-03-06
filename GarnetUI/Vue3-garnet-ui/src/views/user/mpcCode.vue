<template>
  <el-button @click="saveCodeToFile">下载MPC代码到本地</el-button>
  <el-button @click="triggerFileInput">加载本地MPC文件</el-button>
  <el-button @click="showServeMPC">管理服务器MPC文件</el-button>
  <br>
  <br>
  <div class="form-container">
  <el-form  :model="filePost" :rules="rules" label-width="0px" ref="data1">
    <el-form-item prop = "fileName" label="">
      <el-input v-model="filePost.fileName" class="inputlength" placeholder="请填写文件名"/>
    </el-form-item>
  </el-form>
  <el-form  :model="filePost"  label-width="50px" >
    <el-form-item prop = "description" label="">
      <el-input v-model="filePost.description" class="inputlength" placeholder="（可选）请填写文件描述"/>
    </el-form-item>
  </el-form>
  </div>  
  <input type="file" @change="loadFile" ref="fileInput" style="display: none;" /> 
  
  <div class="editor-container" @scroll="syncScroll">
    <div class="line-numbers">
      <div v-for="n in numberOfLines" :key="n" class="line-number">{{ n }}</div>
    </div>
    <textarea
      v-model="code"
      @input="updateLineNumbers"
      @keydown.tab.prevent="handleTab"
      class="code-input"
    ></textarea>
  </div>
  <br>
  <el-button @click="compileCode">对上述代码进行编译</el-button>
  <el-button @click="upLoadFile">上传文件</el-button>
  <div class="output-container">
  <pre>{{ compileOutput }}</pre>
  </div>
  <el-dialog
    v-model="fileDialogVisible"
    title="以下是服务器中的MPC文件"
    width="60%"
    :before-close="fileDialogHandleClose"
  >
    <el-col :span="18" :offset="5">
    <el-table 
    :data="fileTableData" 
    style="width: 100%" 
    highlight-current-row 
    max-height="400"
    >
        <el-table-column prop="id" label="文件id" width="80" />
        <el-table-column prop="name" label="文件名" width="180" />
        <el-table-column prop="description" label="文件描述" width="240"/>
        <el-table-column fixed="right" label="操作" width="105" header-align="center">
        <template v-slot="scope">
          <!-- <el-button
            link
            type="primary"
            @click="loadMPCFile(scope.row.id)"
            size="small"
            >读取</el-button
          > -->
          <el-button
            link
            type="primary"
            @click="deleteMPCFile(scope.row.id)"
            size="small"
            >删除MPC文件</el-button
          >
        </template>
      </el-table-column>
      </el-table>
      </el-col>
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
  </template>
  
  <script lang="ts" setup>
  import { ref, computed } from 'vue';
  import { uploadMpcToServer,compileMpc } from '../../api/user';
  import { getAllModel } from '../../api/locateCompute';
  import request from '../../utils/request'
  const code = ref('');
  const numberOfLines = computed(() => code.value.split('\n').length);

  const rules = {
  fileName:[{required:true,message:'请输入文件名',trigger:'change'}]}

  const syncScroll = (event) => {
  // 同步滚动逻辑
};
const feedbackDialogVisible = ref(false)
const feedbackMessage = ref()
const feedbackDialogClose = () => {
  feedbackDialogVisible.value = false
}

  const updateLineNumbers = () => {
    // 这里可以添加其他当文本变化时需要执行的代码
  };
  
  const saveCodeToFile = () => {
  const blob = new Blob([code.value], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'my-code.mpc';  // 设置文件默认名称和后缀
  document.body.appendChild(link); // 添加到页面
  link.click();
  document.body.removeChild(link); // 移除链接
  URL.revokeObjectURL(url);
};

  const handleTab = (event) => {
    const start = event.target.selectionStart;
    const end = event.target.selectionEnd;
  
    // 设置Tab的缩进空格数，通常是4个空格
    const tabCharacter = '    ';
  
    if (event.shiftKey) {
    // Shift + Tab，减少缩进
    if (code.value.substring(start - 4, start) === tabCharacter) {
      code.value = code.value.substring(0, start - 4) + code.value.substring(start);
      event.target.selectionStart = start - 4;
      event.target.selectionEnd = end - 4;
    }
  } else {
    // Tab，增加缩进
    code.value = code.value.substring(0, start) + tabCharacter + code.value.substring(end);
    event.target.selectionStart = start + tabCharacter.length;
    event.target.selectionEnd = start + tabCharacter.length;
  }
  
    event.preventDefault();
  };

const data1 = ref()
const upLoadFile = async()=>{
  filePost.value.content = code.value;
  await data1.value.validate()
  try {
  await uploadMpcToServer(filePost.value);
  feedbackMessage.value = '上传成功！';
  feedbackDialogVisible.value = true;
} catch (error) {
  if (error.response && error.response.status === 409) {
    feedbackMessage.value = 'MPC文件重名，请修改文件名！';
  } else {
    feedbackMessage.value = '发生错误，请稍后再试！';
  }
  feedbackDialogVisible.value = true;
}
}

const filePost = ref({
//   id:1,
  content: '',
  fileName: '',
  description:''
})


const fileTableData = ref([])
const fileDialogVisible = ref(false)
const showServeMPC = async() => {
  //展示对话框
  fileDialogVisible.value = true
  //展示所有MPC文件
  const res = await getAllModel()
  fileTableData.value = res.data.results
}
const apiResultUrl = ref('/model/mpc/')
const testResultUrl = ref('/model/mpc/')
const deleteMPCFile = async(id) => {
  //拼接并删除
  testResultUrl.value = apiResultUrl.value + id
  await request.delete(testResultUrl.value)
  //重新获取所有MPC文件
  const res = await getAllModel()
  fileTableData.value = res.data.results
  //提示
  feedbackMessage.value='删除成功！'
  feedbackDialogVisible.value = true
}

// const loadMPCFile = async(id) => {
//   //拼接并获取
//   testResultUrl.value = apiResultUrl.value + id
//   const res = await request.get(testResultUrl.value)
//   //将获取的MPC文件内容赋值给code
//   code.value = res.data.content
//   //提示
//   feedbackMessage.value='读取成功！'
//   feedbackDialogVisible.value = true
// }


const fileDialogHandleClose = () => {
  fileDialogVisible.value = false
}

const compileFilePost = ref({
  content: '',
  fileName: '',
  description:'1'
})

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
      await uploadMpcToServer(filePost.value);
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

const tempMpcCompile = ref({
  content:'',
  parameters:''
})
const compileOutput = ref()
const compileCode = async()=>{
    tempMpcCompile.value.content = code.value
    const res = await compileMpc(tempMpcCompile.value)
    compileOutput.value = res.data.out+'\n'+res.data.err
}
const fileInput = ref(null);

const triggerFileInput = () => {
  if(fileInput.value!=null)fileInput.value.click();
};

const loadFile = event => {
  const file = event.target.files[0];
  if (file) {
    const reader = new FileReader();
    reader.onload = e => {
    if(e.target!=null)
      code.value = e.target.result as string;
    };
    reader.readAsText(file);
  }
};
  </script>
  
  <style>
  .editor-container {
    display: flex;
    height: 300px; /* 设置固定高度 */
    overflow-y: auto; /* 超出高度时显示滚动条 */
    overflow-x: auto
  }
  
  .line-numbers {
    padding: 5px;
    background-color: #eee;
    text-align: right;
    user-select: none;
    flex-shrink: 0; /* 防止行号区域缩小 */
    height: 9000px;
  }
  
  .line-number {
    padding: 0 5px;
  }
  
  .code-input {
    flex: 1;
    padding: 5px;
    border: none;
    font-family: monospace;
    line-height: 1.5;
    overflow: hidden; /* 隐藏textarea的滚动条 */
    resize: none; /* 防止用户调整大小 */
    height: 9000px;
  }
  .output-container {
  border: 1px solid #ddd;
  background-color: #f4f4f4;
  padding: 8px;
  margin-top: 20px;
  height: 150px; 
   /* 根据需要调整高度 */
  overflow-y: auto; 
  /* 添加滚动条 */
  font-family: 'Consolas', 'Monaco', 'monospace';
  /* // 设置等宽字体 */
  white-space: 'pre-wrap'; 
  /* 保留空白符和换行 */
  color: #333;
}
.inputlength {
  width: 200px;
}
.form-container {
  display: flex;
  justify-content: flex-start; /* 或者使用其他的对齐方式 */
}

/* 可选：如果需要，可以调整每个表单的宽度 */
.el-form {
  /* flex: 1; 这会使每个表单占据相等的空间 */
}

  </style>
  