import onnx
from onnx import helper, shape_inference
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np

def load_model(model_path):
    model = onnx.load(model_path)
    return model

def save_model(model, model_path):
    onnx.save(model, model_path)

def create_submodel(node, inputs, outputs, initializers, name="submodel"):
    # Create the graph for the submodel
    graph = helper.make_graph(
        [node],
        name,
        inputs,
        outputs,
        initializers
    )
    # Create the model
    model = helper.make_model(graph)
    # Inference the shapes to complete the model
    model = shape_inference.infer_shapes(model)
    return model

def get_tensor_shape(graph, tensor_name):
    # Find the tensor shape in the graph's value_info, input, or output
    for value_info in graph.value_info:
        if value_info.name == tensor_name:
            return [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
    for input_info in graph.input:
        if input_info.name == tensor_name:
            return [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    for output_info in graph.output:
        if output_info.name == tensor_name:
            return [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
    return None

def get_input_sizes(model):
    # Load the ONNX model
    graph = model.graph

    # Initialize dictionaries to store input and output sizes
    tensor_shapes = {}
    node_input_sizes = {}

    # Get initial tensor shapes from inputs and initializers
    for input_info in graph.input:
        tensor_shapes[input_info.name] = get_tensor_shape(graph, input_info.name)
    for initializer in graph.initializer:
        tensor_shapes[initializer.name] = list(numpy_helper.to_array(initializer).shape)

    # Iterate over each node in the graph
    for node in graph.node:
        input_sizes = []
        for input_name in node.input:
            if input_name in tensor_shapes:
                input_sizes.append(tensor_shapes[input_name])
            else:
                input_sizes.append(get_tensor_shape(graph, input_name))
        
        # Store the input sizes for the current node
        node_input_sizes[node.name] = input_sizes

        # Determine the output shapes of the current node and store them
        # Here, you would need to compute or infer the shapes based on the operation
        # Since ONNX doesn't provide a built-in shape inference, you might need to handle this manually or use onnxruntime for shape inference
        for output_name in node.output:
            # Example: assume output shape is the same as the first input shape (this is not generally true)
            if input_sizes:
                tensor_shapes[output_name] = input_sizes[0]

    return node_input_sizes

def split_model_by_node(model_path, input_data = np.random.rand(1, 64, 56, 56).astype(np.float32)):
    model = onnx.load(model_path)
    
    onnx.checker.check_model(model)

    # Initialize the ONNX Runtime session
    session = ort.InferenceSession(model_path)
    # 获取模型的输入和输出节点名称
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape

    # 为模型创建一个带有形状推断的模型
    onnx.checker.check_model(model)
    onnx.helper.strip_doc_string(model)
    onnx.shape_inference.infer_shapes(model)
    
    # 提取模型图
    graph = model.graph

    # 用于存储中间节点的shape信息
    shape_info = {}

    # 遍历所有的节点
    for node in graph.node:
        for output in node.output:
            # 获取节点输出的信息
            value_info = next((vi for vi in graph.value_info if vi.name == output), None)
            if value_info:
                shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
                shape_info[output] = shape

    print(shape_info)
    
    submodels = []
    input_sizes = []
    node_info = []
    
    # Get input sizes of the submodel
    input_size = get_input_sizes(model)
    
    for node in model.graph.node:
        # Determine inputs and outputs for the submodel
        input_names = set(node.input)
        output_names = set(node.output)
        
        # Filter initializers and value infos
        inputs = [vi for vi in model.graph.input if vi.name in input_names]
        outputs = [vi for vi in model.graph.output if vi.name in output_names]
        initializers = [init for init in model.graph.initializer if init.name in input_names]
        
        # Create and save the submodel
        submodel = create_submodel(node, inputs, outputs, initializers, name=node.name)
        submodels.append(submodel)

        # Record the node name and parameters
        node_info.append((node.name, input_size[node.name]))
        
    return submodels, node_info
