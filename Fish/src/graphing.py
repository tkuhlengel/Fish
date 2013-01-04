'''
Created on Oct 22, 2012
\brief Purpose of this module is to automatically output some basic information about 
    a 3D volume based on some statistical analysis and projections of the fish.
\author Trevor Kuhlengel
'''
import numpy as np
import matplotlib as mpl
import multiprocessing as mp
from matplotlib.pylab import figure,hist,subplots,show,axvline
def show_hist(datai, mean1, mean2, avg,title="Histogram", bins=100):
    #mean1,mean2.min(),npdata.max()
    #avg=(mean1+mean2)/2
    fig,ax=subplots(1,1,sharex=True,sharey=True, figsize=(10,10))
    n,bins,patches=ax.hist(datai.flat, bins=bins,range=(mean1,mean2),histtype='bar')
    axvline(x=avg, alpha=0.7, linewidth=3, color='r')
    ax.set_title("histogram")  
    show()
def parallelize(func,Pool=None, *args, **kwargs):
    if Pool is None:
        p=mp.Pool(processes=3)
    elif type(Pool)==type(mp.Pool()):
        p=Pool
    else:
        raise mp.ProcessError("Pool must either be None or an instance of the multiprocessing.Pool() class")
    results=[]
    for i in range(3):
        kwargs["axis"]=i
        results.append(p.apply_async(func, args, kwargs))
    return results
    
def output_image_series(nparray,vertical=True,horizontal=False, figsize=(10,10)):
    #data=np.asanyarray(nparray)
    #dims=len(nparray.shape)
    #if dims>3:
    #    nparray=np.squeeze(nparray)

    
    for i in range(len(nparray)):
        fig=mpl.pyplot.figure(figsize=figsize, dpi=101)
        data=nparray[i]
        if len(data.shape)!=2:
            if len(nparray[i].shape)<2:
                raise Exception("Dimension Error, nparray must be an iterable with 2-dimensional numpy arrays")
            else:
                data=np.squeeze(nparray[i])
            
        elif len(nparray[i].shape)==2:
            data=nparray[i]
        fig=mpl.pyplot.imshow(nparray[i])
        mpl.pyplot.show(fig)        
    
def output_slice_stats(scaledgrid1,axis=0, axis_labels=["Z","Y","X"], savefile=False, 
                       savefileloc='/home/trevor/Documents/Sam29-33_10um_{1}{0}_Statistics_per_slice.png'):
    assert axis<len(np.shape(scaledgrid1)), "Assertion Error: Argument axis must be  \
        less than the number of axes in the data."
    
    #The swap axis (sa) list and the axis label (al) shortcuts for brevity
    sa=[[0,0],[0,1],[0,2]] #Swap axis list
    al=axis_labels
    
    #Do the axis swaps if necessary and move relevant components such as labels
    if axis==0:
        scaledgridview=scaledgrid1

    else:
        print("Swapping axis {1} with axis {0}".format(
                al[sa[axis][0]],al[sa[axis][1]]))
        scaledgridview=np.swapaxes(scaledgrid1,sa[axis][0],sa[axis][1])
        
        #Swap the labels for outputs on the charts
        al[sa[axis][0]],al[sa[axis][1]]=al[sa[axis][1]],al[sa[axis][0]]
    
    #Set the x-axis range
    xaxis=np.arange(len(scaledgridview))
    
    if savefile:
        assert len(savefileloc)>0, "Assertion Error in output_slice_stats: \
        If savefile is true, savefileloc must be a filename/path to save the image."
    
    counter=0
    #Allocate arrays for faster storage
    stdevs=np.zeros(len(scaledgridview))
    diffs=np.zeros(len(scaledgridview))
    varis=np.zeros(len(scaledgridview))
    
    #Calculate the slices the way I intended
    #TODO Rework this so that it's faster
    for slicei in scaledgridview:
        stdevs[counter]=np.std(slicei.flat)
        diffs[counter]=np.ptp(slicei.flat)#slicei.max()-slicei.min()
        varis[counter]=np.var(slicei.flat)
        counter+=1
    #print("Shapes: \nStDevs={}\nDiffs = {}\nVariances = {}".format(
    #np.shape(stdevs),np.shape(diffs),np.shape(varis)))

    
    
    fig=mpl.pyplot.figure(1, figsize=(30,60),dpi=101)
    #ax=mpl.axis.Tick()
    
    #Standard Deviation Plot
    
    sp1=mpl.pyplot.subplot(3,1,1)
    sp1.set_label("Standard Deviation per Slice")
    mpl.pyplot.bar(xaxis,stdevs,rasterized=False,figure=sp1, animated=False,antialiased=True)
    sp1.set_title("Standard Deviation per Slice {1}{0} Plane".format(al[1],al[2]))
    sp1.set_xlabel("Slice Number")
    sp1.set_ylabel("Standard Deviation of Intensity")

    fig.add_subplot(sp1)
    #print("Mean of StDev = {}\nStandard Deviation of StDev = {}".format(np.mean(stdevs),np.std(stdevs)))
    sp2=mpl.pyplot.subplot(3,1,2)

    #Variance Plot
    mpl.pyplot.bar(xaxis,varis, figure=sp2,antialiased=True)
    sp2.set_title("Variance per Slice {1}{0} Plane".format(al[1],al[2]))
    sp2.set_xlabel("Slice Number")
    sp2.set_ylabel("Variance")
    fig.add_subplot(sp2)
    
    #Absolute Difference Plot
    sp3=mpl.pyplot.subplot(3,1,3)
    mpl.pyplot.bar(xaxis,diffs, figure=sp3,antialiased=True)
    sp3.set_xlabel("Slice Number")
    sp3.set_ylabel("Intensity")
    sp3.set_title("Slice Intensity Range {1}{0} Plane".format(al[1],al[2]))
    fig.add_subplot(sp3)
    if savefile:
        mpl.pyplot.savefig(savefileloc.format(al[1],al[2]), dpi=200)
    mpl.pyplot.show(fig)
    del scaledgridview,sp3,sp2,sp1,fig
    return xaxis,stdevs,varis,diffs

def output_projections(scaledgrid,axis=0, _parallel=False, savefile=False, savedpi=200,
                       savefileloc='/home/trevor/Documents/Sam29-33_10um_Projection_along_axis_{}',
                       figsize=(10,30), dpi=101):
    ##! \brief Projects a 3D volume into a 2D plane along a given axis


    axmean=np.mean(scaledgrid,axis)
    axmax=np.amax(scaledgrid,axis)
    axvar=np.var(scaledgrid,axis)
    
    #print("Shapes:\n{}\n{}\n{}".format(np.shape(axmean),np.shape(axmax),np.shape(axvar)))
    fig=mpl.pyplot.figure(1,figsize=figsize,dpi=dpi)
    
    sub1=mpl.pyplot.subplot(1,3,1)
    mpl.pyplot.imshow(axmean,figure=sub1, animated=False)
    sub1.set_title("Mean Projection along axis {}".format(axis))
    #mpl.pyplot.colorbar()
    fig.add_subplot(sub1)
    
    sub2=mpl.pyplot.subplot(1,3,2)
    mpl.pyplot.imshow(axmax,figure=sub2, animated=False)
    sub2.set_title("Maximum Projection along axis {}".format(axis))
    #mpl.pyplot.colorbar()
    fig.add_subplot(sub2)
    
    sub3=mpl.pyplot.subplot(1,3,3)
    mpl.pyplot.imshow(axvar,figure=sub3, animated=False)
    sub3.set_title("Variance Projection along axis {}".format(axis))
    #mpl.pyplot.colorbar()
    fig.add_subplot(sub3)
            
    mpl.pyplot.show(fig)

    if savefile:
        mpl.pyplot.savefig(savefileloc.format(axis), dpi=savedpi)
    return fig,axmean,axmax,axvar

def output_figure(imglist,figsize=(10,30), dpi=101):
    '''
    Outputs a figure from the images in imglist.  Stacks images horizontally across the figure.
    '''
    fig=mpl.pyplot.figure(figsize=figsize,dpi=dpi)
    cols=len(imglist)
    for i in np.arange(cols):
      
        #print("Shapes:\n{}\n{}\n{}".format(np.shape(axmean),np.shape(axmax),np.shape(axvar)))
        
        sub0=mpl.pyplot.subplot(1,3,i)
        mpl.pyplot.imshow(imglist[i],figure=sub0, animated=False)
        sub0.set_title("Maximum Projection along axis {}".format(i))
        mpl.pyplot.colorbar()
        fig.add_subplot(sub0)
    return fig
    
print("Module graphing.py loaded successfully")
if __name__=='__main__':
    import fish
    testfish="/mnt/scratch/tkk3/joinedImages/resized/rec_J29-33_33D_PTAwt_16p2_30mm_w4Dd_xy292_z1047_cubic-0-5_10.0um.nrrd"
    fish=fish.Fish(testfish)
    fish.load()
    