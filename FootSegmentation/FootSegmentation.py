# -*- coding: utf-8 -*-
"""
FootSegmentation - 3D Slicer Extension
AI-powered 3D foot segmentation using ONNX model.

This extension provides automatic segmentation of foot structures
from NIfTI/DICOM volumes using a pre-trained deep learning model.

Authors:
    Cüneyt Özdemir (Siirt University)
    Mehmet Ali Gedik (Kütahya Health Sciences University)

Version: 1.0.0
"""

import os
import logging
import numpy as np
from typing import Optional, Tuple

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
import vtk
import ctk
import qt

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not installed. Please install: pip install onnxruntime")


# =============================================================================
# MODULE DEFINITION
# =============================================================================

class FootSegmentation(ScriptedLoadableModule):
    """
    3D Slicer module for automatic foot segmentation.
    """
    
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        
        self.parent.title = "Foot Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = [
            "Cüneyt Özdemir (Siirt University)",
            "Mehmet Ali Gedik (Kütahya Health Sciences University)"
        ]
        self.parent.helpText = """
        <h3>3D Foot Segmentation</h3>
        <p>This module performs automatic segmentation of 3D foot images using artificial intelligence.</p>
        
        <h4>Usage:</h4>
        <ol>
            <li>Load a volume (NIfTI or DICOM)</li>
            <li>Select it as "Input Volume"</li>
            <li>Click the "Segment" button</li>
            <li>The result will be automatically added to the segmentation node</li>
        </ol>
        
        <p><b>Note:</b> Model loading may take a few seconds on the first run.</p>
        """
        self.parent.acknowledgementText = """
        This module was developed by Cüneyt Özdemir (Siirt University) and 
        Mehmet Ali Gedik (Kütahya Health Sciences University).
        The deep learning model uses 3D U-Net architecture.
        """


# =============================================================================
# WIDGET (UI)
# =============================================================================

class FootSegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """
    Main widget for user interface.
    """
    
    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic = None
        self._updatingGUIFromParameterNode = False
    
    def setup(self):
        """Setup the widget UI."""
        ScriptedLoadableModuleWidget.setup(self)
        
        # Create logic
        self.logic = FootSegmentationLogic()
        
        # =====================================================================
        # Input Section
        # =====================================================================
        inputCollapsibleButton = ctk.ctkCollapsibleButton()
        inputCollapsibleButton.text = "Input Settings"
        self.layout.addWidget(inputCollapsibleButton)
        
        inputFormLayout = qt.QFormLayout(inputCollapsibleButton)
        
        # Input volume selector
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = True
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Select the volume to be segmented")
        inputFormLayout.addRow("Input Volume: ", self.inputSelector)
        
        # =====================================================================
        # Output Section
        # =====================================================================
        outputCollapsibleButton = ctk.ctkCollapsibleButton()
        outputCollapsibleButton.text = "Output Settings"
        self.layout.addWidget(outputCollapsibleButton)
        
        outputFormLayout = qt.QFormLayout(outputCollapsibleButton)
        
        # Output segmentation selector
        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ["vtkMRMLSegmentationNode"]
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip("Segmentation result will be saved here")
        outputFormLayout.addRow("Output Segmentation: ", self.outputSelector)
        
        # =====================================================================
        # Parameters Section
        # =====================================================================
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        parametersCollapsibleButton.collapsed = True
        self.layout.addWidget(parametersCollapsibleButton)
        
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)
        
        # Threshold slider
        self.thresholdSlider = ctk.ctkSliderWidget()
        self.thresholdSlider.singleStep = 0.05
        self.thresholdSlider.minimum = 0.1
        self.thresholdSlider.maximum = 0.9
        self.thresholdSlider.value = 0.5
        self.thresholdSlider.setToolTip("Segmentation threshold value (0.5 recommended)")
        parametersFormLayout.addRow("Threshold: ", self.thresholdSlider)
        
        # Overlap slider
        self.overlapSlider = ctk.ctkSliderWidget()
        self.overlapSlider.singleStep = 0.1
        self.overlapSlider.minimum = 0.1
        self.overlapSlider.maximum = 0.75
        self.overlapSlider.value = 0.5
        self.overlapSlider.setToolTip("Sliding window overlap (higher = better but slower)")
        parametersFormLayout.addRow("Overlap: ", self.overlapSlider)
        
        # Use GPU checkbox
        self.useGPUCheckBox = qt.QCheckBox()
        self.useGPUCheckBox.checked = False
        self.useGPUCheckBox.setToolTip("Use CUDA GPU if available (faster)")
        parametersFormLayout.addRow("Use GPU: ", self.useGPUCheckBox)
        
        # =====================================================================
        # Segment Button
        # =====================================================================
        self.segmentButton = qt.QPushButton("🔍 Start Segmentation")
        self.segmentButton.toolTip = "Start automatic segmentation on the selected volume"
        self.segmentButton.enabled = False
        self.segmentButton.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.layout.addWidget(self.segmentButton)
        
        # =====================================================================
        # Status Section
        # =====================================================================
        statusCollapsibleButton = ctk.ctkCollapsibleButton()
        statusCollapsibleButton.text = "Status"
        self.layout.addWidget(statusCollapsibleButton)
        
        statusFormLayout = qt.QFormLayout(statusCollapsibleButton)
        
        # Status label
        self.statusLabel = qt.QLabel("Ready")
        self.statusLabel.setStyleSheet("color: green; font-weight: bold;")
        statusFormLayout.addRow("Status: ", self.statusLabel)
        
        # Progress bar
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        statusFormLayout.addRow("Progress: ", self.progressBar)
        
        # Model info
        modelPath = self.logic.getModelPath()
        modelStatus = "✓ Loaded" if os.path.exists(modelPath) else "✗ Not Found"
        self.modelLabel = qt.QLabel(modelStatus)
        statusFormLayout.addRow("Model: ", self.modelLabel)
        
        # Add vertical spacer
        self.layout.addStretch(1)
        
        # =====================================================================
        # Connections
        # =====================================================================
        self.segmentButton.clicked.connect(self.onSegmentButton)
        self.inputSelector.currentNodeChanged.connect(self.updateButtonState)
        self.outputSelector.currentNodeChanged.connect(self.updateButtonState)
        
        # Initial update
        self.updateButtonState()
    
    def cleanup(self):
        """Clean up when module is closed."""
        pass
    
    def updateButtonState(self):
        """Update segment button enabled state."""
        inputVolume = self.inputSelector.currentNode()
        self.segmentButton.enabled = inputVolume is not None and ONNX_AVAILABLE
        
        if not ONNX_AVAILABLE:
            self.statusLabel.setText("ERROR: onnxruntime not installed!")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold;")
    
    def onSegmentButton(self):
        """Run segmentation when button clicked."""
        
        inputVolume = self.inputSelector.currentNode()
        outputSegmentation = self.outputSelector.currentNode()
        
        if inputVolume is None:
            slicer.util.errorDisplay("Please select an input volume!")
            return
        
        # Create output segmentation if not selected
        if outputSegmentation is None:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode", 
                f"{inputVolume.GetName()}_Segmentation"
            )
            self.outputSelector.setCurrentNode(outputSegmentation)
        
        # Update status
        self.statusLabel.setText("Segmentation in progress...")
        self.statusLabel.setStyleSheet("color: orange; font-weight: bold;")
        self.progressBar.setValue(0)
        slicer.app.processEvents()
        
        try:
            # Get parameters
            threshold = self.thresholdSlider.value
            overlap = self.overlapSlider.value
            useGPU = self.useGPUCheckBox.checked
            
            # Run segmentation
            self.logic.runSegmentation(
                inputVolume,
                outputSegmentation,
                threshold=threshold,
                overlap=overlap,
                useGPU=useGPU,
                progressCallback=self.updateProgress
            )
            
            self.statusLabel.setText("Completed!")
            self.statusLabel.setStyleSheet("color: green; font-weight: bold;")
            self.progressBar.setValue(100)
            
        except Exception as e:
            self.statusLabel.setText(f"ERROR: {str(e)}")
            self.statusLabel.setStyleSheet("color: red; font-weight: bold;")
            logging.error(f"Segmentation error: {e}")
            import traceback
            traceback.print_exc()
    
    def updateProgress(self, value: int, message: str = ""):
        """Update progress bar."""
        self.progressBar.setValue(value)
        if message:
            self.statusLabel.setText(message)
        slicer.app.processEvents()


# =============================================================================
# LOGIC
# =============================================================================

class FootSegmentationLogic(ScriptedLoadableModuleLogic):
    """
    Segmentation logic using ONNX model.
    """
    
    # Model URL - Will be downloaded from GitHub Releases
    # Update this URL according to your GitHub release
    MODEL_URL = "https://github.com/cuneytozdemir/SlicerFootSegmentation/releases/download/v1.0.0/foot_segmentation.onnx"
    MODEL_FILENAME = "foot_segmentation.onnx"
    
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.session = None
        self.modelLoaded = False
        
        # Model configuration - must match ONNX model input shape (batch, depth, height, width, channels)
        # Based on error analysis: index 2 expects 64, index 3 expects 128
        # So model expects (1, 64, 64, 128, 1) -> patchSize = (depth=64, height=64, width=128)
        self.patchSize = (64, 64, 128)
        self.inputName = None
        self.outputName = None
    
    def getModelPath(self) -> str:
        """Get path to ONNX model file."""
        modulePath = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(modulePath, "Resources", "Models", self.MODEL_FILENAME)
    
    def downloadModel(self, progressCallback=None) -> bool:
        """
        Download ONNX model from GitHub Releases.
        Automatically downloads if model file does not exist.
        """
        import urllib.request
        import ssl
        
        modelPath = self.getModelPath()
        
        # Skip download if model already exists
        if os.path.exists(modelPath):
            logging.info(f"Model already exists: {modelPath}")
            return True
        
        # Create Models directory
        modelsDir = os.path.dirname(modelPath)
        os.makedirs(modelsDir, exist_ok=True)
        
        logging.info(f"Downloading model from: {self.MODEL_URL}")
        
        if progressCallback:
            progressCallback(2, "Downloading model...")
        
        try:
            # SSL context
            context = ssl.create_default_context()
            
            # Download
            urllib.request.urlretrieve(self.MODEL_URL, modelPath)
            
            logging.info(f"Model downloaded successfully: {modelPath}")
            return True
            
        except Exception as e:
            logging.error(f"Model download failed: {e}")
            raise RuntimeError(
                f"Model could not be downloaded. Please check your internet connection.\n"
                f"Manual download: {self.MODEL_URL}\n"
                f"Target location: {modelPath}"
            )
    
    def loadModel(self, useGPU: bool = False):
        """Load ONNX model (automatically downloads if not present)."""
        
        if self.modelLoaded:
            return True
        
        modelPath = self.getModelPath()
        
        # Download model if not present
        if not os.path.exists(modelPath):
            self.downloadModel()
        
        # Setup providers
        if useGPU:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        logging.info(f"Loading ONNX model: {modelPath}")
        self.session = ort.InferenceSession(modelPath, providers=providers)
        
        # Get input/output names
        self.inputName = self.session.get_inputs()[0].name
        self.outputName = self.session.get_outputs()[0].name
        
        self.modelLoaded = True
        
        # Log model input shape for debugging
        input_shape = self.session.get_inputs()[0].shape
        logging.info(f"ONNX model input shape: {input_shape}")
        logging.info(f"ONNX model input name: {self.inputName}")
        logging.info(f"ONNX model output name: {self.outputName}")
        logging.info("ONNX model loaded successfully")
        
        return True
    
    def normalizeVolume(self, volume: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        mean = np.mean(volume)
        std = np.std(volume)
        if std > 0:
            return (volume - mean) / std
        return volume - mean
    
    def createGaussianWeight(self, shape: Tuple[int, int, int]) -> np.ndarray:
        """Create Gaussian weight for patch blending."""
        d, h, w = shape
        z = np.linspace(-1, 1, d)
        y = np.linspace(-1, 1, h)
        x = np.linspace(-1, 1, w)
        zz, yy, xx = np.meshgrid(z, y, x, indexing='ij')
        sigma = 0.25
        gaussian = np.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))
        return gaussian.astype(np.float32)
    
    def slidingWindowInference(self, volume: np.ndarray, overlap: float = 0.5,
                                progressCallback=None) -> np.ndarray:
        """Perform sliding window inference."""
        
        # Normalize
        volume = self.normalizeVolume(volume)
        
        pd, ph, pw = self.patchSize
        d, h, w = volume.shape
        
        # Padding
        pad_d = (pd - d % pd) % pd
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        
        volume_padded = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='constant')
        d_pad, h_pad, w_pad = volume_padded.shape
        
        # Stride
        stride_d = max(1, int(pd * (1 - overlap)))
        stride_h = max(1, int(ph * (1 - overlap)))
        stride_w = max(1, int(pw * (1 - overlap)))
        
        # Initialize
        prediction = np.zeros_like(volume_padded, dtype=np.float32)
        weight_map = np.zeros_like(volume_padded, dtype=np.float32)
        gaussian = self.createGaussianWeight(self.patchSize)
        
        # Patch locations
        patch_locations = []
        for z in range(0, d_pad - pd + 1, stride_d):
            for y in range(0, h_pad - ph + 1, stride_h):
                for x in range(0, w_pad - pw + 1, stride_w):
                    patch_locations.append((z, y, x))
        
        total_patches = len(patch_locations)
        logging.info(f"Total patches: {total_patches}")
        
        # Process patches
        for i, (z, y, x) in enumerate(patch_locations):
            patch = volume_padded[z:z+pd, y:y+ph, x:x+pw]
            patch_input = patch[np.newaxis, ..., np.newaxis].astype(np.float32)
            
            # Run inference
            result = self.session.run([self.outputName], {self.inputName: patch_input})
            pred_patch = result[0][0, ..., 0]
            
            prediction[z:z+pd, y:y+ph, x:x+pw] += pred_patch * gaussian
            weight_map[z:z+pd, y:y+ph, x:x+pw] += gaussian
            
            # Progress callback
            if progressCallback and (i + 1) % 5 == 0:
                progress = int(10 + 80 * (i + 1) / total_patches)
                progressCallback(progress, f"Patch {i+1}/{total_patches}")
        
        # Normalize
        prediction = np.divide(prediction, weight_map, 
                              out=np.zeros_like(prediction), 
                              where=weight_map > 0)
        
        # Remove padding
        prediction = prediction[:d, :h, :w]
        
        return prediction
    
    def runSegmentation(self, inputVolume, outputSegmentation, 
                        threshold: float = 0.5, overlap: float = 0.5,
                        useGPU: bool = False, progressCallback=None):
        """
        Run segmentation on input volume.
        
        Args:
            inputVolume: vtkMRMLScalarVolumeNode
            outputSegmentation: vtkMRMLSegmentationNode
            threshold: Binarization threshold
            overlap: Sliding window overlap
            useGPU: Use GPU if available
            progressCallback: Progress callback function
        """
        
        if progressCallback:
            progressCallback(5, "Loading model...")
        
        # Load model
        self.loadModel(useGPU)
        
        if progressCallback:
            progressCallback(10, "Reading volume...")
        
        # Get volume array
        volumeArray = slicer.util.arrayFromVolume(inputVolume)
        logging.info(f"Volume shape: {volumeArray.shape}")
        
        # Run inference
        prediction = self.slidingWindowInference(volumeArray, overlap, progressCallback)
        
        if progressCallback:
            progressCallback(90, "Creating segmentation...")
        
        # Binarize
        binaryMask = (prediction > threshold).astype(np.uint8)
        
        # Create labelmap volume
        labelmapVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
        slicer.util.updateVolumeFromArray(labelmapVolume, binaryMask)
        
        # Copy geometry from input
        labelmapVolume.SetOrigin(inputVolume.GetOrigin())
        labelmapVolume.SetSpacing(inputVolume.GetSpacing())
        
        # Copy direction
        ijkToRAS = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRAS)
        labelmapVolume.SetIJKToRASMatrix(ijkToRAS)
        
        # Import to segmentation
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolume, outputSegmentation
        )
        
        # Clean up
        slicer.mrmlScene.RemoveNode(labelmapVolume)
        
        # Rename segment
        segmentation = outputSegmentation.GetSegmentation()
        if segmentation.GetNumberOfSegments() > 0:
            segment = segmentation.GetNthSegment(0)
            segment.SetName("Foot")
            segment.SetColor(0.9, 0.2, 0.2)  # Red color
        
        if progressCallback:
            progressCallback(100, "Completed!")
        
        logging.info("Segmentation completed successfully")


# =============================================================================
# TEST
# =============================================================================

class FootSegmentationTest(ScriptedLoadableModuleTest):
    """Test case for FootSegmentation module."""
    
    def setUp(self):
        slicer.mrmlScene.Clear()
    
    def runTest(self):
        self.setUp()
        self.test_FootSegmentation1()
    
    def test_FootSegmentation1(self):
        self.delayDisplay("Starting the test")
        
        # Test logic instantiation
        logic = FootSegmentationLogic()
        
        self.delayDisplay("Test passed")
