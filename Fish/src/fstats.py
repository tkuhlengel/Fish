#!/usr/bin/python2.7
## \brief Module implementing common statistical techniques.
#Created on Oct 23, 2012
#
#\author Trevor Kuhlengel

import numpy as np
from scipy import stats


## \brief Calculates the mode of an array as a left-sided value of a bin count
# \param[in] nparray A numpy array of any shape or structure
# \param[in] bincount Integer or list define the number of bins to use in the
#    histogram when finding the maximum
# \param[in] get_value Boolean denoting whether the maximum value should be
#    included in the result object. Defaults to True
# \param[in] get_count Boolean denoting whether to return the number of items
#    in the greatest bin. Defaults to False
# \note If more than one of get_value or get_count is requested, a list is
#    returned containing them in the order [get_value,get_count]
def mode(nparray, bincount=1000, get_value=True, get_count=False):
    n, bins = np.histogram(nparray, bins=bincount)
    result = []
    sorted_index = n.argsort()
    if get_value:
        result.append(bins[sorted_index[-1]])
    if get_count:
        result.append(n[sorted_index[-1]])
    if len(result) == 1:
        return result[0]
    return result

    #for i in np.arange(len(bins)):
    #    if n[i]==val:
    #        if get_value and get_index:
    #            return bins[i],i
    #        elif get_value:
    #            return bins[i]
    #        elif get_index:


def histogramming(volume, sqSize=10, bins=10)
	#Try to gake a volume histogram and match the histograms to organ surfaces
	for layer in volume:
		binEdges=[]
		hists=[]
		results=[]
		for yslice in range(0,layer.shape[0],sqSize):
			for xslice in range(0, layer.shape[1],sqSize):
				square=layer[yslice:yslice+sqSize,xslice:xslice+sqSize]
				#stats.histogram(square, numbins=bins
				hist,bin_edges=np.histogram(square, bins=bins, density=True)
				hists.append(hist)
				binEdges.append(bin_edges)
				results.append(np.column_stack(hist, bin_edges))
	return results

def compareHists(histograms, bin_edges, groups=[])
	groups=[]
	standards=[]
	
	
	
