/**
 *
 */
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageIOBase.h>
#include "ReadUnknownImageType.cxx"
#include "load.h"

int main( int argc, char **argv) {
	typedef itk::Image<itk::ImageIOBase::IOComponentType>			ImageType;
	//typedef itk::ImageFileReader<ImageType>							ReaderType;
	//typedef itk::GradientMagnitudeImageFilter<ImageType, ImageType>	FilterType;
	std::string inputFfilename = argv[1];
	//ReaderType::Pointer reader = ReaderType::New();
	cout<<loadUnknownImageType(&filename, );
	//Define the volume type to pass along

	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(inputFilename.c_str(), itk::ImageIOFactory::ReadMode);

	return 0;
}
