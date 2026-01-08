# -*- coding: utf-8 -*-
"""
FootSegmentation - 3D Slicer Extension
AI-powered 3D foot segmentation using ONNX model.

This extension provides automatic segmentation of foot structures
from NIfTI/DICOM volumes using a pre-trained deep learning model.

Author: [Cüneyt ÖZDEMİR, Mehmet Ali GEDİK]
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
            "Cüneyt ÖZDEMİR (Siirt University),Mehmet Ali GEDİK (Kutahya University of Health Sciences)"]
        self.parent.helpText = """
        <h3>3D Foot Segmentation</h3>
        <p>This module performs automatic segmentation on 3D foot images using artificial intelligence.</p>
        
        <h4>Usage:</h4>
        <ol>
            <li>Load a volume (NIfTI or DICOM)</li>
            <li>Select it as the "Input Volume"</li>
            <li>Click the "Segment" button</li>
            <li>The result will be automatically added to the segmentation node</li>
        </ol>
        
        <p><b>Note:</b> Model loading may take a few seconds on first run.</p>
        """
        self.parent.acknowledgementText = """
        This module was developed by Cüneyt ÖZDEMİR (Siirt University) and Mehmet Ali GEDİK (Kutahya University of Health Sciences).
        The deep learning model is based on a 3D U-Net architecture.
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
        
        self.thresholdSlider = ctk.ctkSliderWidget()
        self.thresholdSlider.singleStep = 0.05
        self.thresholdSlider.minimum = 0.1
        self.thresholdSlider.maximum = 0.9
        self.thresholdSlider.value = 0.5
        self.thresholdSlider.setToolTip("Segmentation threshold (0.5 recommended)")
        parametersFormLayout.addRow("Threshold: ", self.thresholdSlider)
        
        self.overlapSlider = ctk.ctkSliderWidget()
        self.overlapSlider.singleStep = 0.1
        self.overlapSlider.minimum = 0.1
        self.overlapSlider.maximum = 0.75
        self.overlapSlider.value = 0.5
        self.overlapSlider.setToolTip("Sliding window overlap (higher = better but slower)")
        parametersFormLayout.addRow("Overlap: ", self.overlapSlider)
        
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
        self.layout.addWidget(self.segmentButton)
        
        # =====================================================================
        # Status Section
        # =====================================================================
        statusCollapsibleButton = ctk.ctkCollapsibleButton()
        statusCollapsibleButton.text = "Status"
        self.layout.addWidget(statusCollapsibleButton)
        
        statusFormLayout = qt.QFormLayout(statusCollapsibleButton)
        
        self.statusLabel = qt.QLabel("Ready")
        statusFormLayout.addRow("Status: ", self.statusLabel)
        
        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        statusFormLayout.addRow("Progress: ", self.progressBar)
        
        modelPath = self.logic.getModelPath()
        modelStatus = "✓ Loaded" if os.path.exists(modelPath) else "✗ Not Found"
        self.modelLabel = qt.QLabel(modelStatus)
        statusFormLayout.addRow("Model: ", self.modelLabel)
        
        self.layout.addStretch(1)
        
        self.segmentButton.clicked.connect(self.onSegmentButton)
        self.inputSelector.currentNodeChanged.connect(self.updateButtonState)
        self.outputSelector.currentNodeChanged.connect(self.updateButtonState)
        
        self.updateButtonState()
    
    def updateButtonState(self):
        inputVolume = self.inputSelector.currentNode()
        self.segmentButton.enabled = inputVolume is not None and ONNX_AVAILABLE
        
        if not ONNX_AVAILABLE:
            self.statusLabel.setText("ERROR: onnxruntime is not installed!")
    
    def onSegmentButton(self):
        inputVolume = self.inputSelector.currentNode()
        outputSegmentation = self.outputSelector.currentNode()
        
        if inputVolume is None:
            slicer.util.errorDisplay("Please select an input volume!")
            return
        
        if outputSegmentation is None:
            outputSegmentation = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode",
                f"{inputVolume.GetName()}_Segmentation"
            )
            self.outputSelector.setCurrentNode(outputSegmentation)
        
        self.statusLabel.setText("Running segmentation...")
        self.progressBar.setValue(0)
        
        try:
            self.logic.runSegmentation(
                inputVolume,
                outputSegmentation,
                threshold=self.thresholdSlider.value,
                overlap=self.overlapSlider.value,
                useGPU=self.useGPUCheckBox.checked,
                progressCallback=self.updateProgress
            )
            self.statusLabel.setText("Completed!")
            self.progressBar.setValue(100)
        except Exception as e:
            self.statusLabel.setText(f"ERROR: {str(e)}")
    
    def updateProgress(self, value: int, message: str = ""):
        self.progressBar.setValue(value)
        if message:
            self.statusLabel.setText(message)
        slicer.app.processEvents()
