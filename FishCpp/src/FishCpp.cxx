/**
 *
 */
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageIOBase.h>
#include "load.h"

int main( int argc, char **argv) {
	typedef itk::Image<itk::ImageIOBase::IOComponentType>			ImageType;
	//typedef itk::ImageFileReader<ImageType>							ReaderType;
	//typedef itk::GradientMagnitudeImageFilter<ImageType, ImageType>	FilterType;
	std::string filename = argv[1];
	//ReaderType::Pointer reader = ReaderType::New();
	itk::Object data = loadUnknownImageType(&filename);

	return 0;
}
