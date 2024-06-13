import onnx
from onnx import helper, shape_inference

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

def split_model_by_node(model):
    submodels = []
    
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
    
    return submodels

if __name__=='__main__':
    # Example usage
    model_path = "example.onnx"
    model = load_model(model_path)

    submodels = split_model_by_node(model)

    # Save submodels
    for i, submodel in enumerate(submodels):
        save_model(submodel, f"submodel_{i+1}.onnx")
