import argparse
import os
import platform
import sys
import time
import warnings
import glob
import cv2
from pathlib import Path

import pandas as pd
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel
from utils.general import (
    LOGGER,
    Profile,
    check_img_size,
    check_requirements,
    check_version,
    colorstr,
    file_size,
    get_default_args,
    print_args,
    url2file,
)
from utils.torch_utils import select_device, smart_inference_mode

MACOS = platform.system() == "Darwin"  # macOS environment



def export_formats():
    r"""
    Returns a DataFrame of supported YOLOv5 model export formats and their properties.

    Returns:
        pandas.DataFrame: A DataFrame containing supported export formats and their properties. The DataFrame
        includes columns for format name, CLI argument suffix, file extension or directory name, and boolean flags
        indicating if the export format supports training and detection.

    Examples:
        ```python
        formats = export_formats()
        print(f"Supported export formats:\n{formats}")
        ```

    Notes:
        The DataFrame contains the following columns:
        - Format: The name of the model format (e.g., PyTorch, TorchScript, ONNX, etc.).
        - Include Argument: The argument to use with the export script to include this format.
        - File Suffix: File extension or directory name associated with the format.
        - Supports Training: Whether the format supports training.
        - Supports Detection: Whether the format supports detection.
    """
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlpackage", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def try_export(inner_func):
    """
    Log success or failure, execution time, and file size for YOLOv5 model export functions wrapped with @try_export.

    Args:
        inner_func (Callable): The model export function to be wrapped by the decorator.

    Returns:
        Callable: The wrapped function that logs execution details. When executed, this wrapper function returns either:
            - Tuple (str | torch.nn.Module): On success — the file path of the exported model and the model instance.
            - Tuple (None, None): On failure — None values indicating export failure.

    Examples:
        ```python
        @try_export
        def export_onnx(model, filepath):
            # implementation here
            pass

        exported_file, exported_model = export_onnx(yolo_model, 'path/to/save/model.onnx')
        ```

    Notes:
        For additional requirements and model export formats, refer to the
        [Ultralytics YOLOv5 GitHub repository](https://github.com/ultralytics/ultralytics).
    """
    inner_args = get_default_args(inner_func)

    def outer_func(*args, **kwargs):
        """Logs success/failure and execution details of model export functions wrapped with @try_export decorator."""
        prefix = inner_args["prefix"]
        try:
            with Profile() as dt:
                f, model = inner_func(*args, **kwargs)
            LOGGER.info(f"{prefix} export success ✅ {dt.t:.1f}s, saved as {f} ({file_size(f):.1f} MB)")
            return f, model
        except Exception as e:
            LOGGER.info(f"{prefix} export failure ❌ {dt.t:.1f}s: {e}")
            return None, None

    return outer_func


@try_export
def export_onnx(model, im, file, opset, dynamic, simplify, prefix=colorstr("ONNX:")):
    """
    Export a YOLOv5 model to ONNX format with dynamic axes support and optional model simplification.

    Args:
        model (torch.nn.Module): The YOLOv5 model to be exported.
        im (torch.Tensor): A sample input tensor for model tracing, usually the shape is (1, 3, height, width).
        file (pathlib.Path | str): The output file path where the ONNX model will be saved.
        opset (int): The ONNX opset version to use for export.
        dynamic (bool): If True, enables dynamic axes for batch, height, and width dimensions.
        simplify (bool): If True, applies ONNX model simplification for optimization.
        prefix (str): A prefix string for logging messages, defaults to 'ONNX:'.

    Returns:
        tuple[pathlib.Path | str, None]: The path to the saved ONNX model file and None (consistent with decorator).

    Raises:
        ImportError: If required libraries for export (e.g., 'onnx', 'onnx-simplifier') are not installed.
        AssertionError: If the simplification check fails.

    Notes:
        The required packages for this function can be installed via:
        ```
        pip install onnx onnx-simplifier onnxruntime onnxruntime-gpu
        ```

    Example:
        ```python
        from pathlib import Path
        import torch
        from models.experimental import attempt_load
        from utils.torch_utils import select_device

        # Load model
        weights = 'yolov5s.pt'
        device = select_device('')
        model = attempt_load(weights, map_location=device)

        # Example input tensor
        im = torch.zeros(1, 3, 640, 640).to(device)

        # Export model
        file_path = Path('yolov5s.onnx')
        export_onnx(model, im, file_path, opset=12, dynamic=True, simplify=True)
        ```
    """
    check_requirements("onnx>=1.12.0")
    import onnx

    LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__}...")
    f = str(file.with_suffix(".onnx"))

    output_names = ["output0", "output1"] if isinstance(model, SegmentationModel) else ["output0"]
    if dynamic:
        dynamic = {"images": {0: "batch", 2: "height", 3: "width"}}  # shape(1,3,640,640)
        if isinstance(model, SegmentationModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)
            dynamic["output1"] = {0: "batch", 2: "mask_height", 3: "mask_width"}  # shape(1,32,160,160)
        elif isinstance(model, DetectionModel):
            dynamic["output0"] = {0: "batch", 1: "anchors"}  # shape(1,25200,85)

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
        dynamic_axes=dynamic or None,
    )

    # Checks
    model_onnx = onnx.load(f)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    # Metadata
    d = {"stride": int(max(model.stride)), "names": model.names}
    for k, v in d.items():
        meta = model_onnx.metadata_props.add()
        meta.key, meta.value = k, str(v)
    onnx.save(model_onnx, f)

    # Simplify
    if simplify:
        try:
            cuda = torch.cuda.is_available()
            check_requirements(("onnxruntime-gpu" if cuda else "onnxruntime", "onnxslim"))
            import onnxslim

            LOGGER.info(f"{prefix} slimming with onnxslim {onnxslim.__version__}...")
            model_onnx = onnxslim.slim(model_onnx)
            onnx.save(model_onnx, f)
        except Exception as e:
            LOGGER.info(f"{prefix} simplifier failure: {e}")
    return f, model_onnx


@try_export
def export_engine(
    model, im, file, half, dynamic, simplify, workspace=4, verbose=False, cache="", int8=False,prefix=colorstr("TensorRT:")
):

    assert im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. `python export.py --device 0`"
    try:
        import tensorrt as trt
    except Exception:
        if platform.system() == "Linux":
            check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
        import tensorrt as trt

    if trt.__version__[0] == "7":  # TensorRT 7 handling https://github.com/ultralytics/yolov5/issues/6012
        grid = model.model[-1].anchor_grid
        model.model[-1].anchor_grid = [a[..., :1, :1, :] for a in grid]
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
        model.model[-1].anchor_grid = grid
    else:  # TensorRT >= 8
        check_version(trt.__version__, "8.0.0", hard=True)  # require tensorrt>=8.0.0
        export_onnx(model, im, file, 12, dynamic, simplify)  # opset 12
    onnx = file.with_suffix(".onnx")

    LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
    is_trt10 = int(trt.__version__.split(".")[0]) >= 10  # is TensorRT >= 10
    assert onnx.exists(), f"failed to export ONNX file: {onnx}"
    f = file.with_suffix(".engine")  # TensorRT engine file
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE

    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    if is_trt10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)
    else:  # TensorRT versions 7, 8
        config.max_workspace_size = workspace * 1 << 30
    if cache:  # enable timing cache
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        buf = Path(cache).read_bytes() if Path(cache).exists() else b""
        timing_cache = config.create_timing_cache(buf)
        config.set_timing_cache(timing_cache, ignore_mismatch=True)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError(f"failed to load ONNX file: {onnx}")

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    for inp in inputs:
        LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
    for out in outputs:
        LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

    if dynamic:
        if im.shape[0] <= 1:
            LOGGER.warning(f"{prefix} WARNING ⚠️ --dynamic model requires maximum --batch-size argument")
        profile = builder.create_optimization_profile()
        for inp in inputs:
            profile.set_shape(inp.name, (1, *im.shape[1:]), (max(1, im.shape[0] // 2), *im.shape[1:]), im.shape)
        config.add_optimization_profile(profile)

    if int8 and builder.platform_has_fast_int8:

        class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calibration_files, batch_size=8, input_shape=(3, 640, 1280), cache_file="calib.cache"):
                super().__init__()
                self.cache_file = cache_file
                self.batch_size = batch_size
                self.data = calibration_files
                self.current_index = 0
                self.input_shape = input_shape  # (C, H, W)

            def get_batch_size(self):
                return self.batch_size

            def get_batch(self, names):
                if self.current_index + self.batch_size > len(self.data):
                    return None

                batch = []
                for i in range(self.batch_size):
                    img = cv2.imread(self.data[self.current_index + i])
                    img = cv2.resize(img, (self.input_shape[2], self.input_shape[1]))  # (W, H)
                    img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
                    batch.append(img)

                batch = np.ascontiguousarray(batch)
                tensor = torch.from_numpy(batch).cuda()  # GPU 올리기
                self.current_index += self.batch_size

                # Torch tensor의 GPU 메모리 포인터 반환
                return [int(tensor.data_ptr())]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    return open(self.cache_file, "rb").read()

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        config.set_flag(trt.BuilderFlag.INT8)
        calibration_files = glob.glob("./calib/*")
        assert calibration_files is not None, "❌ int8 exporting needs calibration_files"
        config.int8_calibrator = TorchEntropyCalibrator(
            calibration_files, batch_size=8, input_shape=im.shape[1:], cache_file="calib.cache"
        )

        LOGGER.info(f"{prefix} building INT8 engine as {f}")

    elif builder.platform_has_fast_fp16 and half:
        LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")
        config.set_flag(trt.BuilderFlag.FP16)

    else:
        LOGGER.info(f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and half else 32} engine as {f}")

    # if builder.platform_has_fast_fp16 and half:
    #     config.set_flag(trt.BuilderFlag.FP16)

    build = builder.build_serialized_network if is_trt10 else builder.build_engine
    with build(network, config) as engine, open(f, "wb") as t:
        t.write(engine if is_trt10 else engine.serialize())
    if cache:  # save timing cache
        with open(cache, "wb") as c:
            c.write(config.get_timing_cache().serialize())
    return f, None


@try_export
def export_saved_model(
    model,
    im,
    file,
    dynamic,
    tf_nms=False,
    agnostic_nms=False,
    topk_per_class=100,
    topk_all=100,
    iou_thres=0.45,
    conf_thres=0.25,
    keras=False,
    prefix=colorstr("TensorFlow SavedModel:"),
):
    """
    Export a YOLOv5 model to the TensorFlow SavedModel format, supporting dynamic axes and non-maximum suppression
    (NMS).

    Args:
        model (torch.nn.Module): The PyTorch model to convert.
        im (torch.Tensor): Sample input tensor with shape (B, C, H, W) for tracing.
        file (pathlib.Path): File path to save the exported model.
        dynamic (bool): Flag to indicate whether dynamic axes should be used.
        tf_nms (bool, optional): Enable TensorFlow non-maximum suppression (NMS). Default is False.
        agnostic_nms (bool, optional): Enable class-agnostic NMS. Default is False.
        topk_per_class (int, optional): Top K detections per class to keep before applying NMS. Default is 100.
        topk_all (int, optional): Top K detections across all classes to keep before applying NMS. Default is 100.
        iou_thres (float, optional): IoU threshold for NMS. Default is 0.45.
        conf_thres (float, optional): Confidence threshold for detections. Default is 0.25.
        keras (bool, optional): Save the model in Keras format if True. Default is False.
        prefix (str, optional): Prefix for logging messages. Default is "TensorFlow SavedModel:".

    Returns:
        tuple[str, tf.keras.Model | None]: A tuple containing the path to the saved model folder and the Keras model instance,
        or None if TensorFlow export fails.

    Notes:
        - The method supports TensorFlow versions up to 2.15.1.
        - TensorFlow NMS may not be supported in older TensorFlow versions.
        - If the TensorFlow version exceeds 2.13.1, it might cause issues when exporting to TFLite.
          Refer to: https://github.com/ultralytics/yolov5/issues/12489

    Example:
        ```python
        model, im = ...  # Initialize your PyTorch model and input tensor
        export_saved_model(model, im, Path("yolov5_saved_model"), dynamic=True)
        ```
    """
    # YOLOv5 TensorFlow SavedModel export
    try:
        import tensorflow as tf
    except Exception:
        check_requirements(f"tensorflow{'' if torch.cuda.is_available() else '-macos' if MACOS else '-cpu'}<=2.15.1")

        import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

    from models.tf import TFModel

    LOGGER.info(f"\n{prefix} starting export with tensorflow {tf.__version__}...")
    if tf.__version__ > "2.13.1":
        helper_url = "https://github.com/ultralytics/yolov5/issues/12489"
        LOGGER.info(
            f"WARNING ⚠️ using Tensorflow {tf.__version__} > 2.13.1 might cause issue when exporting the model to tflite {helper_url}"
        )  # handling issue https://github.com/ultralytics/yolov5/issues/12489
    f = str(file).replace(".pt", "_saved_model")
    batch_size, ch, *imgsz = list(im.shape)  # BCHW

    tf_model = TFModel(cfg=model.yaml, model=model, nc=model.nc, imgsz=imgsz)
    im = tf.zeros((batch_size, *imgsz, ch))  # BHWC order for TensorFlow
    _ = tf_model.predict(im, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    inputs = tf.keras.Input(shape=(*imgsz, ch), batch_size=None if dynamic else batch_size)
    outputs = tf_model.predict(inputs, tf_nms, agnostic_nms, topk_per_class, topk_all, iou_thres, conf_thres)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.trainable = False
    keras_model.summary()
    if keras:
        keras_model.save(f, save_format="tf")
    else:
        spec = tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        m = tf.function(lambda x: keras_model(x))  # full model
        m = m.get_concrete_function(spec)
        frozen_func = convert_variables_to_constants_v2(m)
        tfm = tf.Module()
        tfm.__call__ = tf.function(lambda x: frozen_func(x)[:4] if tf_nms else frozen_func(x), [spec])
        tfm.__call__(im)
        tf.saved_model.save(
            tfm,
            f,
            options=tf.saved_model.SaveOptions(experimental_custom_gradients=False)
            if check_version(tf.__version__, "2.6")
            else tf.saved_model.SaveOptions(),
        )
    return f, keras_model



@smart_inference_mode()
def run(
    data=ROOT / "data/coco128.yaml",  # 'dataset.yaml path'
    weights=ROOT / "yolov5s.pt",  # weights path
    imgsz=(640, 640),  # image (height, width)
    batch_size=1,  # batch size
    device="cpu",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    include=("torchscript", "onnx"),  # include formats
    half=False,  # FP16 half-precision export
    inplace=False,  # set YOLOv5 Detect() inplace=True
    keras=False,  # use Keras
    optimize=False,  # TorchScript: optimize for mobile
    int8=False,  # CoreML/TF INT8 quantization
    per_tensor=False,  # TF per tensor quantization
    dynamic=False,  # ONNX/TF/TensorRT: dynamic axes
    cache="",  # TensorRT: timing cache path
    simplify=False,  # ONNX: simplify model
    mlmodel=False,  # CoreML: Export in *.mlmodel format
    opset=12,  # ONNX: opset version
    verbose=False,  # TensorRT: verbose log
    workspace=4,  # TensorRT: workspace size (GB)
    nms=False,  # TF: add NMS to model
    agnostic_nms=False,  # TF: add agnostic NMS to model
    topk_per_class=100,  # TF.js NMS: topk per class to keep
    topk_all=100,  # TF.js NMS: topk for all classes to keep
    iou_thres=0.45,  # TF.js NMS: IoU threshold
    conf_thres=0.25,  # TF.js NMS: confidence threshold
):
    t = time.time()
    include = [x.lower() for x in include]  # to lowercase
    fmts = tuple(export_formats()["Argument"][1:])  # --include arguments
    flags = [x in include for x in fmts]
    assert sum(flags) == len(include), f"ERROR: Invalid --include {include}, valid --include arguments are {fmts}"
    jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle = flags  # export booleans
    file = Path(url2file(weights) if str(weights).startswith(("http:/", "https:/")) else weights)  # PyTorch weights

    # Load PyTorch model
    device = select_device(device)
    if half:
        assert device.type != "cpu" or coreml, "--half only compatible with GPU export, i.e. use --device 0"
        assert not dynamic, "--half not compatible with --dynamic, i.e. use either --half or --dynamic but not both"
    model = attempt_load(weights, device=device, inplace=True, fuse=True)  # load FP32 model

    # Checks
    imgsz *= 2 if len(imgsz) == 1 else 1  # expand
    if optimize:
        assert device.type == "cpu", "--optimize not compatible with cuda devices, i.e. use --device cpu"

    # Input
    gs = int(max(model.stride))  # grid size (max stride)
    imgsz = [check_img_size(x, gs) for x in imgsz]  # verify img_size are gs-multiples
    ch = next(model.parameters()).size(1)  # require input image channels
    im = torch.zeros(batch_size, ch, *imgsz).to(device)  # image size(1,3,320,192) BCHW iDetection

    # Update model
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = inplace
            m.dynamic = dynamic
            m.export = True

    for _ in range(2):
        y = model(im)  # dry runs
    if half and not coreml:
        im, model = im.half(), model.half()  # to FP16
    shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {file} with output shape {shape} ({file_size(file):.1f} MB)")

    # Exports
    f = [""] * len(fmts)  # exported filenames
    warnings.filterwarnings(action="ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
    if engine:  # TensorRT required before ONNX
        f[1], _ = export_engine(model, im, file, half, dynamic, simplify, workspace, verbose, cache, int8)
    if onnx:
        f[2], _ = export_onnx(model, im, file, opset, dynamic, simplify)

    if any((saved_model, pb, tflite, edgetpu, tfjs)):  # TensorFlow formats
        assert not tflite or not tfjs, "TFLite and TF.js models must be exported separately, please pass only one type."
        assert not isinstance(model, ClassificationModel), "ClassificationModel export to TF formats not yet supported."
        f[5], s_model = export_saved_model(
            model.cpu(),
            im,
            file,
            dynamic,
            tf_nms=nms or agnostic_nms or tfjs,
            agnostic_nms=agnostic_nms or tfjs,
            topk_per_class=topk_per_class,
            topk_all=topk_all,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            keras=keras,
        )

    # Finish
    f = [str(x) for x in f if x]  # filter out '' and None
    if any(f):
        cls, det, seg = (isinstance(model, x) for x in (ClassificationModel, DetectionModel, SegmentationModel))  # type
        det &= not seg  # segmentation models inherit from SegmentationModel(DetectionModel)
        dir = Path("segment" if seg else "classify" if cls else "")
        h = "--half" if half else ""  # --half FP16 inference arg
        s = (
            "# WARNING ⚠️ ClassificationModel not yet supported for PyTorch Hub AutoShape inference"
            if cls
            else "# WARNING ⚠️ SegmentationModel not yet supported for PyTorch Hub AutoShape inference"
            if seg
            else ""
        )
        LOGGER.info(
            f"\nExport complete ({time.time() - t:.1f}s)"
            f"\nResults saved to {colorstr('bold', file.parent.resolve())}"
            f"\nDetect:          python {dir / ('detect.py' if det else 'predict.py')} --weights {f[-1]} {h}"
            f"\nValidate:        python {dir / 'val.py'} --weights {f[-1]} {h}"
            f"\nPyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', '{f[-1]}')  {s}"
            f"\nVisualize:       https://netron.app"
        )
    return f  # return list of exported files/dirs


def parse_opt(known=False):
    """
    Parse command-line options for YOLOv5 model export configurations.

    Args:
        known (bool): If True, uses `argparse.ArgumentParser.parse_known_args`; otherwise, uses `argparse.ArgumentParser.parse_args`.
                      Default is False.

    Returns:
        argparse.Namespace: Object containing parsed command-line arguments.

    Example:
        ```python
        opts = parse_opt()
        print(opts.data)
        print(opts.weights)
        ```
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=ROOT / "data/coco.yaml", help="dataset.yaml path")
    parser.add_argument("--weights", nargs="+", type=str, default=ROOT / "yolov5s.pt", help="model.pt path(s)")
    parser.add_argument("--imgsz", "--img", "--img-size", nargs="+", type=int, default=[640, 1280], help="image (h, w)")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size")
    parser.add_argument("--device", default="0", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument("--inplace", action="store_true", help="set YOLOv5 Detect() inplace=True")
    parser.add_argument("--keras", action="store_true", help="TF: use Keras")
    parser.add_argument("--optimize", action="store_true", help="TorchScript: optimize for mobile")
    parser.add_argument("--int8", action="store_true", help="CoreML/TF/OpenVINO INT8 quantization")
    parser.add_argument("--per-tensor", action="store_true", help="TF per-tensor quantization")
    parser.add_argument("--dynamic", action="store_true", help="ONNX/TF/TensorRT: dynamic axes")
    parser.add_argument("--cache", type=str, default="", help="TensorRT: timing cache file path")
    parser.add_argument("--simplify", action="store_true", help="ONNX: simplify model")
    parser.add_argument("--mlmodel", action="store_true", help="CoreML: Export in *.mlmodel format")
    parser.add_argument("--opset", type=int, default=15, help="ONNX: opset version")
    parser.add_argument("--verbose", action="store_true", help="TensorRT: verbose log")
    parser.add_argument("--workspace", type=int, default=4, help="TensorRT: workspace size (GB)")
    parser.add_argument("--nms", action="store_true", help="TF: add NMS to model")
    parser.add_argument("--agnostic-nms", action="store_true", help="TF: add agnostic NMS to model")
    parser.add_argument("--topk-per-class", type=int, default=100, help="TF.js NMS: topk per class to keep")
    parser.add_argument("--topk-all", type=int, default=100, help="TF.js NMS: topk for all classes to keep")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="TF.js NMS: IoU threshold")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="TF.js NMS: confidence threshold")
    parser.add_argument(
        "--include",
        nargs="+",
        default=["engine"],
        help="torchscript, onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle",
    )
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    """Run(**vars(opt))  # Execute the run function with parsed options."""
    for opt.weights in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
