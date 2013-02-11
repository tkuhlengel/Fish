/*
 * import.h
 *
 *  Created on: Jan 23, 2013
 *      Author: trevor
 */

#ifndef IMPORT_H_
#define IMPORT_H_
#include <itkRawImageIO.h>
#include <itkImage.h>
#include <itkImageFileReader.h>

//void loadRaw(fstream &input );

template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);





#endif /* IMPORT_H_ */
