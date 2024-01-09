<template>
  <el-button @click="saveCodeToFile">保存代码到本地</el-button>
  <el-button @click="triggerFileInput">加载本地文件</el-button>
  <br>
  <el-text>
    在此处可以选择本地文件并上传到服务器：
  </el-text>
  <input type="file" ref="fileInput" @change="handleFileChange" />
  <input type="file" @change="loadFile" ref="fileInput" style="display: none;" />
  <el-button @click="uploadFile">上传文件</el-button>
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
  <div class="output-container">
  <pre>{{ compileOutput }}</pre>
  </div>
  </template>
  
  <script lang="ts" setup>
  import { ref, computed } from 'vue';
  import { uploadMpcToServer,compileMpc } from '../../api/user';
  const code = ref('');
  const numberOfLines = computed(() => code.value.split('\n').length);

  const syncScroll = (event) => {
  // 同步滚动逻辑
};

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

  


const filePost = ref({
//   id:1,
  content: '',
  fileName: '',
  description:'1'
})

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
  height: 100px; 
   /* 根据需要调整高度 */
  overflow-y: auto; 
  /* 添加滚动条 */
  font-family: 'Consolas', 'Monaco', 'monospace';
  /* // 设置等宽字体 */
  white-space: 'pre-wrap'; 
  /* 保留空白符和换行 */
  color: #333;
}
  </style>
  