import request from '@/utils/request'
//注册接口
export const userRegistService = ({ username, password }) => {
  let res = request.post('/register/', {
    username,
    password
  })
  return res
}
//登录接口
export const userLoginService = ({ username, password }) => {
  let res = request.post('/login/', {
    username,
    password
  })
  return res
}
export const userLinkServer = ({ ip, port }) => {
  request.post('/link/metadata/send/', {
    ip,
    port
  })
}

//上传mpc文件接口
export const uploadMpcToServer = ({ content, fileName, description }) => {
  let res= request.post('/model/mpc/string/', {
    content, fileName, description
  })
  return res
}
//编译mpc文件
export const compileMpc = ({content,parameters }) => {
  let res= request.post('/model/mpc/compile/', {
    content,parameters
  })
  return res
}

export const getConnectedServer = () => request.get('/model/servers/')
