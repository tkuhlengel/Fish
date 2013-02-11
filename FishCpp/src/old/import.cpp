/*
 * import.cpp
 *
 *  Created on: Jan 23, 2013
 *      Author: trevor
 */
#include "import.h"


template<typename TImageType>
void ReadFile(std::string filename, typename TImageType::Pointer image){
  typedef itk::ImageFileReader<TImageType> ReaderType;
 typename ReaderType::Pointer reader = ReaderType::New();


  reader->SetFileName(filename);
  reader->Update();

  image->Graft(reader->GetOutput());
}
