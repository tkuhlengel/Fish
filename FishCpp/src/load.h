/**
 * \file load.h
 * \brief Code to load and import image files of arbitrary dimensions
 * \author Trevor Kuhlengel <tkuhlengel@gmail.com>
 *
 *

 *
 * \section Description
 * 	Code focused on importing and loading image files into ITK formats for processing, using
 * load.h
 *
 *  Created on: Feb 6, 2013
 *      Author: trevor
 */

#ifndef LOAD_H_
#define LOAD_H_
#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageIOBase.h>
#include <itkNrrdImageIO.h>

#include <iostream>
//#include <map>

//typedef std::map<std::pair<int, int>, int> Dict;
//typedef Dict::const_iterator It;

/** \brief Load a file with an appropriate header automatically into an itk::Image
 *
 *
 *
 */
template<typename TImageType>
static int loadUnknownImageType(std::string *inputFilename,typename TImageType::Pointer image);


template<typename TImageType>
static void ReadFile(std::string filename, typename TImageType::Pointer image);


#endif /* LOAD_H_ */
