/**
 * \file load.cpp
 * \author Trevor Kuhlengel <tkuhlengel@gmail.com>
 * \brief Loads Image files
 */

#include "load.h"

template<typename TImageType>
static typename TImageType::Pointer* loadUnknownImageType(std::string *inputFilename){



	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	itk::ImageIOBase::Pointer imageIO =
			itk::ImageIOFactory::CreateImageIO(
					inputFilename->c_str(), itk::ImageIOFactory::ReadMode);

	imageIO->SetFileName(*inputFilename);
	imageIO->ReadImageInformation();

	//This is where PIXEL COMPONENT TYPE is declared
	const ScalarPixelType pixelType = imageIO->GetComponentType();

	//Attempt to generically define an image
	typedef itk::Image<itk::ImageIOBase::IOComponentType,3> ImageType;
	ImageType::Pointer image = ImageType::New();

	//Read the image file and store it in the pointer
	ReadFile<ImageType>(*inputFilename, image);
	std::cout<<"Successfully imported file "<<inputFilename<<std::endl;

	/*
	switch (imageIO->GetPixelType())
	{
		case itk::ImageIOBase::COVARIANTVECTOR:
			typedef itk::Image<itk::ImageIOBase::IOComponentType,3> ImageType;

			ImageType::Pointer image = ImageType::New();
			ReadFile<ImageType>(inputFilename, image);

			break;
		case itk::ImageIOBase::SCALAR:

			typedef itk::Image<itk::ImageIOBase::IOComponentType, 3> ImageType;
			ImageType::Pointer image = ImageType::New();
			ReadFile<ImageType>(inputFilename, image);
			break;

	default:
		std::cerr << "Pixel Type ("
				<< imageIO->GetComponentTypeAsString(pixelType)
				<< ") not supported. Exiting." << std::endl;
		return -1;
	}
	*/
	return image;
}

template<typename TImageType>
void ReadFile(std::string filename, typename TImageType::Pointer image)
{
	typedef itk::ImageFileReader<TImageType> ReaderType;
	typename ReaderType::Pointer reader = ReaderType::New();

	reader->SetFileName(filename);
	reader->Update();

	image->Graft(reader->GetOutput());
}

itk::Image::Pointer loadNrrdFile(std::string *inputFilename){
	typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

	//itk::ImageIOBase::Pointer imageIO =	itk::NrrdImageIO::CreateImageIO(inputFilename->c_str(), itk::ImageIOFactory::ReadMode);
}


