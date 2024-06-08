import taso
import onnx

#Build DNN model
graph = taso.new_graph()
input = graph.new_input(dims=(1,32,10,10))
w0 = graph.new_weight(dims=(128,32,1,1))
w10 = graph.new_weight(dims=(32,32,1,1))
w11 = graph.new_weight(dims=(32,32,3,3))
w12 = graph.new_weight(dims=(128,32,1,1))

path = input
path1 = graph.conv2d(input=path, weight=w0, strides=(1,1), padding="SAME")
path1 = graph.relu(path1)

path2 = input
path2 = graph.conv2d(input=path2, weight=w10, strides=(1,1), padding="SAME")
path2 = graph.conv2d(input=path2, weight=w11, strides=(1,1), padding="SAME")
path2 = graph.conv2d(input=path2, weight=w12, strides=(1,1), padding="SAME")
path2 = graph.relu(path2)

output = graph.add(path1, path2)

#Optimize DNN model
onnx_model = taso.export_onnx(graph)
onnx.save(onnx_model, "example.onnx")

new_graph = taso.optimize(graph, alpha=1e9, input_size=(1,32,10,10), inMPL=True)

onnx_model = taso.export_onnx(new_graph)
onnx.save(onnx_model, "example_opt.onnx")