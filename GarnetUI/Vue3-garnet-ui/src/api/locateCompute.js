import request from '@/utils/request'
//创建任务接口
export const userCreateLocateTask = ({
  taskName,
  mpc_parameters,
  protocol_parameters,
  pN,
  status,
  description,
  userid,
  mpc,
  protocol
}) => {
  let res = request.post('/task/local/model/', {
    taskName,
    mpc_parameters,
    protocol_parameters,
    pN,
    status,
    description,
    userid,
    mpc,
    protocol
  })
  // console.log(res)
  return res
}
export const getAllLocatetask = () => request.get('/task/local/model/')

export const getAllFiles = () => request.get('/model/userdata/')

export const getAllProtocol = () => request.get('/model/protocol/')

export const getAllModel = () => request.get('/model/mpc/')

export const userPostData = ({ content, description, userID, fileName }) => {
  request.post('/model/userdata/string/', {
    content,
    description,
    userID,
    fileName
  })
}

// const taskdata = [
//   {
//     index: 0,
//     data: 0,
//     task: 0
//   }
// ]
// const tableData = []

export const userFileToTask = (taskdata) => {
  request.post('/task/local/data/', taskdata)
}
