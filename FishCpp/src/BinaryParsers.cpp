//============================================================================
// Name        : BinaryParsers.cpp
// Author      : Trevor Kuhlengel
// Version     :
// Copyright   : Trevor Kuhlengel and Penn State University 2013
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "import.h"
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageIOBase.h>



using namespace std;
template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);
int main(int argc, char *argv[])
{
  if(argc < 2)
    {
    std::cerr << "Required: filename" << std::endl;

    return EXIT_FAILURE;
    }
  std::string inputFilename = argv[1];

  typedef itk::ImageIOBase::IOComponentType ScalarPixelType;

  itk::ImageIOBase::Pointer imageIO =
        itk::ImageIOFactory::CreateImageIO(
            inputFilename.c_str(), itk::ImageIOFactory::ReadMode);

  imageIO->SetFileName(inputFilename);
  imageIO->ReadImageInformation();
  const ScalarPixelType pixelType = imageIO->GetComponentType();
  std::cout << "Pixel Type is " << imageIO->GetComponentTypeAsString(pixelType) // 'double'
            << std::endl;
  const size_t numDimensions =  imageIO->GetNumberOfDimensions();
  std::cout << "numDimensions: " << numDimensions << std::endl; // '2'

  std::cout << "component size: " << imageIO->GetComponentSize() << std::endl; // '8'
  std::cout << "pixel type (string): " << imageIO->GetPixelTypeAsString(imageIO->GetPixelType()) << std::endl; // 'vector'
  std::cout << "pixel type: " << imageIO->GetPixelType() << std::endl; // '5'


  switch (pixelType)
  {
    case itk::ImageIOBase::COVARIANTVECTOR:
      typedef itk::Image<unsigned char, 3> ImageType;
      ImageType::Pointer image = ImageType::New();
      ReadFile<ImageType>(inputFilename, image);
      break;

    case itk::ImageIOBase::
      typedef itk::Image<unsigned char, 3> ImageType;
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

  return EXIT_SUCCESS;
}
template<typename TImageType>
void ReadFile(std::string filename, typename TImageType::Pointer image){
  typedef itk::ImageFileReader<TImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(filename);
  reader->Update();

  image->Graft(reader->GetOutput());
}

