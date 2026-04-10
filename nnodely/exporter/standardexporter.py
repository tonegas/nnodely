import os, torch

from nnodely.visualizer import EmptyVisualizer
from nnodely.exporter.emptyexporter import EmptyExporter
from nnodely.exporter.reporter import Reporter
from nnodely.exporter.export import (
    save_model,
    load_model,
    export_python_model,
    export_pythononnx_model,
    export_onnx_model,
    import_python_model,
    onnx_inference,
)
from nnodely.support.utils import check, enforce_types

from nnodely.support.logger import logging, nnLogger

log = nnLogger(__name__, logging.INFO)


class StandardExporter(EmptyExporter):
    @enforce_types
    def __init__(
        self,
        workspace: str | None = None,
        visualizer: EmptyVisualizer | None = None,
        *,
        save_history: bool = False,
    ):
        super().__init__(workspace, visualizer, save_history)

    def getWorkspace(self):
        return self.workspace_folder if hasattr(self, "workspace_folder") else "."

    def saveTorchModel(self, model, name="net", model_folder=None):
        file_name = name + ".pt"
        model_path = (
            os.path.join(self.getWorkspace(), file_name)
            if model_folder is None
            else os.path.join(model_folder, file_name)
        )
        # TODO check if the folder exist
        torch.save(model.state_dict(), model_path)
        self.visualizer.saveModel("Torch Model", model_path)

    def loadTorchModel(self, model, name="net", model_folder=None):
        file_name = name + ".pt"
        model_path = (
            os.path.join(self.getWorkspace(), file_name)
            if model_folder is None
            else os.path.join(model_folder, file_name)
        )
        check(
            os.path.exists(model_path),
            FileNotFoundError,
            f"The model {name} it is not found in the folder {model_folder}",
        )
        model.load_state_dict(torch.load(model_path, weights_only=True))
        self.visualizer.loadModel("Torch Model", model_path)
        # TODO Update the model parameters....

    def saveModel(self, model_def, name="net", model_folder=None):
        # Combine the folder path and file name to form the complete file path
        model_folder = self.getWorkspace() if model_folder is None else model_folder
        # Specify the JSON file name
        file_name = name + ".json"
        # Combine the folder path and file name to form the complete file path
        model_path = os.path.join(model_folder, file_name)
        save_model(model_def, model_path)
        self.visualizer.saveModel("JSON Model", model_path)

    def loadModel(self, name="net", model_folder=None):
        # Combine the folder path and file name to form the complete file path
        model_folder = self.getWorkspace() if model_folder is None else model_folder
        model_def = None
        try:
            file_name = name + ".json"
            model_path = os.path.join(model_folder, file_name)
            model_def = load_model(model_path)
            self.visualizer.loadModel("JSON Model", model_path)
        except Exception as e:
            check(
                False,
                FileNotFoundError,
                f"The file {model_path} it is not found or not conformed.\n Error: {e}",
            )
        return model_def

    def exportPythonModel(self, model_def, model, name="net", model_folder=None):
        file_name = name + ".py"
        model_path = (
            os.path.join(self.getWorkspace(), file_name)
            if model_folder is None
            else os.path.join(model_folder, file_name)
        )
        ## Export to python file
        export_python_model(model_def.getJson(), model, model_path)
        self.visualizer.exportModel("Python Torch Model", model_path)

    def importPythonModel(self, name="net", model_folder=None):
        try:
            model_folder = self.getWorkspace() if model_folder is None else model_folder
            model = import_python_model(name, model_folder)
            self.visualizer.importModel(
                "Python Torch Model", os.path.join(model_folder, name + ".py")
            )
        except Exception as e:
            model = None
            check(
                False,
                FileNotFoundError,
                f"The model {name} it is not found in the folder {model_folder}.\nError: {e}",
            )
        return model

    def exportONNX(
        self,
        model_def,
        model,
        inputs_order=None,
        outputs_order=None,
        name="net",
        model_folder=None,
    ):
        if inputs_order is None:
            log.info(
                f"The inputs order for the export is not specified, the order will set equal to {set(model_def['Inputs'].keys())}."
            )
        elif set(inputs_order) != set(model_def["Inputs"].keys()):
            raise ValueError(
                f"The inputs are not the same as the model inputs {set(model_def['Inputs'].keys())}."
            )
        if outputs_order is None:
            log.info(
                f"The outputs order for the export is not specified, the order will set equal to {set(model_def['Outputs'].keys())}"
            )
        elif set(outputs_order) != set(model_def["Outputs"].keys()):
            log.info(
                f"The outputs are not the same as the model outputs {set(model_def['Outputs'].keys())}."
            )
        file_name = name + ".py"
        model_folder = (
            os.path.join(self.getWorkspace(), "onnx")
            if model_folder is None
            else model_folder
        )
        os.makedirs(model_folder, exist_ok=True)
        model_path = os.path.join(model_folder, file_name)
        onnx_python_model_path = model_path.replace(".py", "_onnx.py")
        onnx_model_path = model_path.replace(".py", ".onnx")
        ## Export to python file (onnx compatible)
        export_python_model(model_def, model, model_path)
        self.visualizer.exportModel("Python Torch Model", model_path)
        export_pythononnx_model(
            model_def, model_path, onnx_python_model_path, inputs_order, outputs_order
        )
        self.visualizer.exportModel("Python Onnx Torch Model", onnx_python_model_path)
        ## Export to onnx file (onnx compatible)
        model = import_python_model(file_name.replace(".py", "_onnx"), model_folder)
        export_onnx_model(
            model_def,
            model,
            onnx_model_path,
            inputs_order,
            outputs_order,
            name=name + "_onnx",
        )
        self.visualizer.exportModel("Onnx Model", onnx_model_path)

    def onnxInference(self, inputs, name: str = "net", model_folder: str | None = None):
        model_folder = (
            os.path.join(self.getWorkspace(), "onnx")
            if model_folder is None
            else model_folder
        )
        file_name = name + ".onnx"
        onnx_model_path = os.path.join(model_folder, file_name)
        check(
            os.path.exists(onnx_model_path),
            FileNotFoundError,
            f"The model {file_name} it is not found in the folder {model_folder}",
        )
        return onnx_inference(inputs, onnx_model_path)

    def exportReport(self, n4m, name="net", model_folder=None):
        # Combine the folder path and file name to form the complete file path
        model_folder = self.getWorkspace() if model_folder is None else model_folder
        # Specify the JSON file name
        file_name = name + ".pdf"
        # Combine the folder path and file name to form the complete file path
        report_path = os.path.join(model_folder, file_name)
        reporter = Reporter(n4m)
        reporter.exportReport(report_path)
        self.visualizer.exportReport("Training Results", report_path)
