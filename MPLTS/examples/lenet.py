import taso as ts
import onnx

def lenet_block(graph, input):
    w1 = graph.new_weight(dims=(6, input.dim(1), 5, 5))
    t = graph.conv2d(input=input, weight=w1, strides=(1, 1), padding="VALID")
    t = graph.relu(t)
    t = graph.maxpool2d(input=t, kernels=(2,2), strides=(2,2), padding="VALID")
    
    w2 = graph.new_weight(dims=(16, t.dim(1), 5, 5))
    t = graph.conv2d(input=t, weight=w2, strides=(1, 1), padding="VALID")
    t = graph.relu(t)
    t = graph.maxpool2d(input=t, kernels=(2,2), strides=(2,2), padding="VALID")

    t = graph.reshape(t, shape=(1, 16*4*4))

    w3 = graph.new_weight(dims=(16*4*4, 120))
    t = graph.matmul(t, w3)
    t = graph.relu(t)

    w4 = graph.new_weight(dims=(120, 84))
    t = graph.matmul(t, w4)
    t = graph.relu(t)

    w5 = graph.new_weight(dims=(84, 10))
    t = graph.matmul(t, w5)

    return t

graph = ts.new_graph()
input = graph.new_input(dims=(1, 1, 28, 28))
output = lenet_block(graph, input)

onnx_model = ts.export_onnx(graph)
onnx.save(onnx_model, "lenet.onnx")

new_graph = ts.optimize(graph, alpha=1.3, budget=1000, input_size=(1, 1, 28, 28), inMPL=True)


onnx_model = ts.export_onnx(new_graph)
onnx.save(onnx_model, "lenet_opt.onnx")
