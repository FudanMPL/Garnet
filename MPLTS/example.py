import taso
import onnx

#Build DNN model
graph = taso.new_graph()
input = graph.new_input(dims=(1,32,10,10))
w11 = graph.new_weight(dims=(32,32,3,3))
w12 = graph.new_weight(dims=(32,32,1,1))
w21 = graph.new_weight(dims=(32,32,3,3))
w22 = graph.new_weight(dims=(32,32,1,1))
w31 = graph.new_weight(dims=(32,32,1,1))
input = graph.relu(input)
path1 = graph.conv2d(input=input, weight=w11, strides=(1,1), padding="SAME")
path1 = graph.conv2d(input=path1, weight=w12, strides=(1,1), padding="SAME")
path1 = graph.relu(path1)
path2 = graph.conv2d(input=input, weight=w21, strides=(1,1), padding="SAME")
path2 = graph.conv2d(input=path2, weight=w22, strides=(1,1), padding="SAME")
path2 = graph.relu(path2)
path3 = graph.conv2d(input=input, weight=w31, strides=(1,1), padding="SAME")
path3 = graph.relu(path3)
output = graph.add(path1, path2)
output = graph.add(output, path3)

#Optimize DNN model

new_graph = taso.optimize(graph, alpha=2, input_size=(1,32,10,10))

onnx_model = taso.export_onnx(graph)
onnx.save(onnx_model, "example.onnx")
onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "example_opt.onnx")