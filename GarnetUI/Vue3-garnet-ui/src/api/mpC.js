import request from '@/utils/request'
//创建任务接口
export const userCreateTask = ({
  taskName,
  mpc_parameters,
  protocol_parameters,
  pN,
  part,
  baseport,
  status,
  description,
  mpc,
  protocol,
  data
}) => {
  request.post('/task/remote/model/', {
    taskName,
    mpc_parameters,
    protocol_parameters,
    pN,
    part,

    baseport,
    status,
    description,
    mpc,
    protocol,
    data
  })
}
export const getAlltask = () => request.get('/task/remote/model/')

export const getAllFiles = () => request.get('/model/userdata/')

export const getAllServers = () => request.get('/model/servers/all/')
export const userPostData = ({ content, description, userID, fileName }) => {
  request.post('/model/userdata/string/', {
    content,
    description,
    userID,
    fileName
  })
}
