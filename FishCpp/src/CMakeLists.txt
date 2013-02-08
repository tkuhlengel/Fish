
cmake_minimum_required(VERSION 2.8)

project(FishCpp)

find_package(ITK REQUIRED)
#include(${ITK_USE_FILE})
#if (ITKVtkGlue_LOADED)
#  find_package(VTK REQUIRED)
#  include(${VTK_USE_FILE})
#endif()
IF (ITK_FOUND)
	INCLUDE(${ITK_USE_FILE})
ENDIF(ITK_FOUND)

ADD_EXECUTABLE(FishCpp FishCpp.cxx)
TARGET_LINK_LIBRARIES(FishCpp ITKCommon ITKIO)

#include(${ITK_USE_FILE})
#if (ITKVtkGlue_LOADED)
#  find_package(VTK REQUIRED)
#  include(${VTK_USE_FILE})
#endif(ITKVtkGlue_LOADED)
#add_executable(ReadUnknownImageType MACOSX_BUNDLE ReadUnknownImageType.cpp)


#if( "${ITK_VERSION_MAJOR}" LESS 4 )
#  target_link_libraries(ReadUnknownImageType ITKReview ${ITK_LIBRARIES})
#else( "${ITK_VERSION_MAJOR}" LESS 4 )
#  target_link_libraries(ReadUnknownImageType ${ITK_LIBRARIES})
#endif( "${ITK_VERSION_MAJOR}" LESS 4 )

