cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(FootSegmentation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/cuneytozdemir/FootSegmentation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Cüneyt ÖZDEMİR (Siirt University),Mehmet Ali GEDİK (Kutahya University of Health Sciences)")
set(EXTENSION_DESCRIPTION "AI-powered automatic 3D foot segmentation using deep learning")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/cuneytozdemir/FootSegmentation/main/FootSegmentation/Resources/Icons/FootSegmentation.png")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "") # Specified as a list
set(EXTENSION_STATUS "Beta")

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(FootSegmentation)

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

