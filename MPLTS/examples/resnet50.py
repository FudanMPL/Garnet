import taso as ts
import onnx

def resnet_block(graph, input, strides, out_channels):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1),3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(4*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*4):
        w4 = graph.new_weight(dims=(out_channels*4,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))

graph = ts.new_graph()
input = graph.new_input(dims=(1,64,56,56))
t = graph.relu(input)
for i in range(3):
    t = resnet_block(graph, t, (1,1), 64)
strides = (2,2)
for i in range(4):
    t = resnet_block(graph, t, strides, 128)
    strides = (1,1)
strides = (2,2)
for i in range(6):
    t = resnet_block(graph, t, strides, 256)
    strides = (1,1)
strides = (2,2)
for i in range(3):
    t = resnet_block(graph, t, strides, 512)
    strides = (1,1)

# new_graph = ts.optimize(graph, alpha=1e9, budget=1000, input_size=(1,64,56,56))
new_graph = ts.optimize(graph, alpha=1e9, budget=1000, input_size=(1,64,56,56), inMPL=True)

onnx_model = ts.export_onnx(graph)
onnx.save(onnx_model, "resnet50.onnx")
onnx_model = ts.export_onnx(new_graph)
onnx.save(onnx_model, "resnet50_opt.onnx")
