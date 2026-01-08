#-----------------------------------------------------------------------------
# FootSegmentation Module
# Authors: Cuneyt Ozdemir (Siirt University), Mehmet Ali Gedik (Kutahya Health Sciences University)
#-----------------------------------------------------------------------------
set(MODULE_NAME FootSegmentation)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/Models/foot_segmentation.onnx
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )
