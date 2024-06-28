import taso
import onnx

d = 32

#Build DNN model
graph = taso.new_graph()
input1 = graph.new_input(dims=(1,d,10,10))
input2 = graph.new_input(dims=(1,d,10,10))

w00 = graph.new_weight(dims=(d,d,3,3))
w01 = graph.new_weight(dims=(d,d,1,1))
w10 = graph.new_weight(dims=(d,d,5,5))
w11 = graph.new_weight(dims=(d,d,1,1))

path1 = input1
path1 = graph.conv2d(input=path1, weight=w00, strides=(1,1), padding="SAME")
path1 = graph.conv2d(input=path1, weight=w01, strides=(1,1), padding="SAME")

path2 = input2
path2 = graph.conv2d(input=path2, weight=w10, strides=(1,1), padding="SAME")
path2 = graph.conv2d(input=path2, weight=w11, strides=(1,1), padding="SAME")

output = graph.add(path1, path2)
output = graph.relu(output)

onnx_model = taso.export_onnx(graph)
onnx.save(onnx_model, "example.onnx")

# graph = taso.optimize(graph, alpha=1e9, input_size=(1,32,10,10))
graph = taso.optimize(graph, alpha=1e9, input_size=(1,d,10,10), inMPL=True)

onnx_model = taso.export_onnx(graph)
onnx.save(onnx_model, "example_opt.onnx")