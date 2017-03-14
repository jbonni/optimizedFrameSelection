import numpy as np
import h5py
import datetime
import operator
from functools import reduce
import os
import random
import math









TRAIN_RATIO = 0.9



"""
!!!!!!!!ERRORS ENCOUNTERED!!!!!!!!
    -array de 1 : a = np.array(1.2312)
    a.shape
    Out[5]: ()


"""



"""
********TO DO********
    - Add path selection for reading & writing
    - save and load the quantity of items in the file
    - Manage different size databases
    - getbatch(training/test)

    - New class datahandler, this class will become hdf5handler and inherit
        from datahandler class or datahandler will load hdf5 handler methods

    - Really slow to save!! add csv and other methods
"""




class dataHandler():
    """
    ##############
    dataHandler
    ##############
    description:
      Class to handle data in and out of an external hdf5 file with h5py.
      Can be used to generate a dataset or read from an existing one.

    example:
        #datahandler object
        dh = dataHandler()

        #fake data
        fake1=np.random.randint(255,size = [7056])
        fake2=np.random.randint(255,size = [4,4])
        fake3=np.random.randint(255,size = [512])
        fake4=np.random.randint(255,size = [28,28,28])

        #save loop
        for _ in range(40001):
            dh.addData(fake1,fake2,fake3,fake4)
        del fake1, fake2, fake3, fake4

        #Empty the buffer and save it
        dh.saveData()

        #load data
        load1, _            = dh.load(395)
        loadList, indexList = dh.load([1,602,332,999,20000])
        loadRange           = dh.loadRange(20,400)
    """
    def __init__(self, fileType = "hdf5"):

        if not isinstance(fileType, str):
            print("ERROR : fileType must be a string")
            raise

        #Max index in the dataset
        self.maxDataIndex = 0
        #Size of the save slice
        self.sliceSize = 20000
        #length of the flatten array with the data
        self.dataLength = 0
        self.dataShape = []

        #Buffer used for single add in memory. This is to avoid to
        #write in the hdf5 file for every new item (painfully slow)
        self.buffer = {}

        self.fileType = fileType

        #filename with timestamp.
        self.fileName = str(os.getcwd()) +"\\test_{}.{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
            self.fileType)





    def addData(self, *data):
        """
        @@@@@@@@@@
        addData
        @@@@@@@@@@

        description:
          Add data to the dataHandler object. When the buffer is full, the
          function will make a call to saveData to save the buffer in the
          external hdf5 file.

        args:
          *data : all the data that will be saved in the hdf5 file. Must be
                  numpy arrays. Can be any dimensions except 0.

        returns:
          NA

        examples:
          for _ in range(100000):
            dh.addData(fakeData1, fakeData2, fakeData3)
        """
        #flatten the data to a 1D array
        flat, shape = self.flattenData(*data)

        #datatype error occured
        if isinstance(flat, bool):
            return

        #first add to the file, note the shape of the array
        if self.dataLength == 0:
            self.dataLength = sum(shape)
            self.dataShape = shape
            self.initBuffer()
        else:
            #For now if the shape is different, cancel operations
            #Possible add to the future: create new datasets when different
            #input sizes.
            if shape != self.dataShape:
                print("DIFFERENT DATA SHAPE: ABORT WRITE")
                raise


        #if buffer full, save to hdf5 file
        if self.buffIndex >= self.sliceSize:
            self.saveData()
        #assign data to the buffer
        self.buffer["data"][self.buffIndex] = flat

        self.buffIndex += 1






    #NOT READY
    def loadBatch(self, batchType = "training"):
        try:
            #first pass or dataset modified
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                if batchType == "training":
                    batch = f["batch/" + str(self.batchIndex)]
                    self.batchIndex += 1
                    return batch
                elif batchType == "test":
                    return f["batch/testBatch"]

        except OSError:
                print("ERROR: File does not exists")
                return

    #NOT READY
    def createBatch(self, batchSize):
        self.batchIndex = 0

        try:
            #first pass or dataset modified
            with h5py.File(self.fileName, "a",  libver='latest') as f:
#                f.create_group("batch")
                qty = f["flatData/quantity"][0]
                fullList = random.sample(range(0, qty),qty)

#                if qty%batchSize != 0 :
#                    randSample = random.sample(range(0, qty%batchSize),
#                                                   qty%batchSize)
#                    fullList += randSample

                f["batch/list"] = fullList

                for i in range(0,math.ceil(qty/batchSize)):
                    start = i*batchSize

                    #training##Ne devrait jamais dépasser vu le 0.9. mais catcher l'erreur
                    if TRAIN_RATIO * qty < start:
                        batchList = fullList[start: start + batchSize]
                        f["batch/"+ str(self.batchIndex)] = \
                          self.load(batchList)
                        self.batchIndex += 1
                    #test
                    else:
                        f["batch/test"] = self.load(fullList[start:])
                        break



        except OSError:
            print("ERROR: File does not exists")
            return






#    def loadBatch(self, batchSize, batchType = "training"):
#        """
#        @@@@@@@@@@
#        batch
#        @@@@@@@@@@
#
#        description:
#
#
#        args:
#
#
#        returns:
#          NA
#
#        examples:
#
#        """
#        try:
#            if batchType == "training":
#                batchList = self.trainList[self.trainPointer:\
#                                           self.trainPointer+batchSize]
#                self.trainPointer += batchSize
#                return self.load(batchList)
#            elif batchType == "test":
#                return self.load(self.testList)
#            else:
#                print("ERROR : invalid batchType")
#
#        except AttributeError:
#            try:
#                #first pass or dataset modified
#                with h5py.File(self.fileName, "r",  libver='latest') as f:
#                    fullList = random.sample(range(0, f["flatData/quantity"][0]),
#                                             f["flatData/quantity"][0])
#            except OSError:
#                print("ERROR: File does not exists")
#                return
#            #create two list for training and test
#            self.trainList = fullList[0:int(0.9*len(fullList))]
#            self.testList = fullList[int(0.9*len(fullList)):]
#            self.trainPointer = 0
#            #retry the batch
#            return self.batch(batchSize, batchType)






    def flattenData(self,*data):
        """
        @@@@@@@@@@
        flattenData
        @@@@@@@@@@

        description:
          Receives the data as separated elements in the function call and
          returns the same information as a single 1D numpy array.

        args:
          *data : data to be flatten to an array.

        returns:
          dataArray : The array with the data.
          dataShape : the shape of the new array. Can be used to indicate the
                      separations in the new array

        examples:
          d1 = np.ones([153])
          d2 = np.zeros([4,4])
          d3 = np.ones([1])
          d4 = np.zeros([1,6,6,1])
          flat,shape = dh.flattenData(d1,d2,d3,d4)
              #flat = flat array of 206 -> [d1,d2,d3,d4]
              #shape = [153, 16, 1, 36]
          flat,shape = dh.flattenData(d1,d2,d3,d4,[1,2,3])
              #flat & shape = False: datatype not supported
          flat,shape = dh.flattenData(d1,d2,d3,d4,(1,2,3))
              #flat & shape = False: datatype not supported
          flat,shape = dh.flattenData(d1,d2,d3,d4,1)
              #flat & shape = False: datatype not supported
        """
        if len(data) == 0:
            print("NO DATA")
            return False, False

        #variables to contain the flatten data and shape as a list
        dataArray = np.zeros(0)
        dataShape = []

        try:
            for element in data:
                #already 1d array
                if len(element.shape) == 1:
                    dataArray = np.append(dataArray, element)
                    dataShape += [element.shape[0]]
                #reshape to a 1d array
                else:
                    new_shape = reduce(operator.mul,element.shape,1)
                    dataShape += [new_shape]
                    dataArray = np.append(dataArray,
                                          np.reshape(element,
                                                     new_shape))
        except AttributeError:
            print("WRONG DATA TYPE, EXPECTED NUMPY ARRAYS: CANCEL OPERATION")
            raise

        return dataArray, dataShape



    def initBuffer(self):
        """
        @@@@@@@@@@
        initBuffer
        @@@@@@@@@@

        description:
          Initialize the buffer to empty arrays.

        args:
          TBC

        returns:
          NA

        examples:
          dh.initBuffer()
        """
        self.buffer["data"] = np.zeros([self.sliceSize,self.dataLength])
        self.buffIndex = 0




    def load(self, index_or_list):
        """
        @@@@@@@@@@
        load
        @@@@@@@@@@

        description:
          loads a single or a list element from the hdf5 file.

        args:
          index_or_list : index of a single item or list.

        returns:
          data : the data requested. If a list was requested, the data will
                 be sorted. Will return false if the range is out of bound.
          index_or_list : returns the index first requested. this is useful
                          if the list has been sorted.

        examples:
          returnedData , _ = dh.load(20) #returns element at index 20.
          returnedData , _ = dh.loadRange([20,60,200]) #returns elements at
                                                       #index 20, 60, 200.
          returnedData , sortedList = dh.loadRange([60,200,20]) #same as above.
          if isinstance(returnedData,bool):
              print("index off bound")
        """
        #the load require a sorted list
        if isinstance(index_or_list, list):
          index_or_list.sort()

        if self.fileType == "hdf5":
            #Open the file in read mode and returns the data
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                try:
                    data = f["flatData/data"][index_or_list]
                except (KeyError, ValueError):
                    #invalid index in
                    print("INVALID INDEXES, LOAD CANCELLED")
                    data = False
            return data#, index_or_list
        else:
            print("fileType not configured")
            raise


    def loadRange(self, start, end):
        """
        @@@@@@@@@@
        loadRange
        @@@@@@@@@@

        description:
          loads a range of elements element from the hdf5 file. If start is
          larger than end, the two elements will be swapped and the range
          returned anyway.

        args:
          start : First elements of the range
          end : Last element of the range

        returns:
          data : the data requested in the specified range. will return False
                 if the range is out of bound.

        examples:
          returnedData = dh.loadRange(20,60)
          returnedData = dh.loadRange(60,20) #same as 20,60
          returnedData = dh.LoadRange(400000:400002)
          if isinstance(returnedData,bool):
              print("range off bound")
        """
        #swap start is bigger than end
        if start>end:
            start, end = end, start

        if self.fileType == "hdf5":
            #Open the file in read mode and returns the data
            with h5py.File(self.fileName, "r",  libver='latest') as f:
                try:
                    data = f["flatData/data"][start:end]

                    if data.shape[0] < end - start:
                        print("INVALID RANGE, LOAD CANCELLED")
                        data = False
                except KeyError:
                    #invalid range
                    print("INVALID RANGE, LOAD CANCELLED")
                    data = False
        elif self.fileType == "csv":
            pass
        else:
            print("fileType not configured")
            return False
        return data


    def saveData(self, dataType = "hdf5"):
        """
        @@@@@@@@@@
        saveData
        @@@@@@@@@@

        description:
          saves the buffer content to the hdf5 file. Will resize the dataset
          to the shape of the buffer.

        args:
          NA

        returns:
          NA

        examples:
          dh.saveData()
        """
        #if saveData is called with buffer not full
        if self.buffIndex < self.sliceSize:
            if self.buffIndex == 0:
                print("CANNOT SAVE: BUFFER EMPTY. SAVE CANCELLED")
                return
            resize = self.buffIndex
        else:
            resize = self.sliceSize

        if self.fileType == "hdf5":
            with h5py.File(self.fileName, "a",  libver='latest') as f:
                #for first write, create group
                if self.maxDataIndex == 0:
                    grp = f.create_group("flatData")
                    grp.create_dataset("data",
                                       data=self.buffer["data"],
                                       compression="lzf",
                                       maxshape=(None, self.dataLength))
                    grp.create_dataset("quantity",
                                       data=[resize])
                #if first group created, resize and add new data
                else:
                    f["flatData/data"].resize(
                        f["flatData/data"].shape[0]+resize,axis=0)
                    f["flatData/data"][-resize:] = self.buffer["data"][:resize]
                    f["flatData/quantity"][0] = resize + \
                                                    f["flatData/quantity"][0]
        else:
            print("fileType not configured")
            raise

        self.initBuffer()
        self.maxDataIndex += resize

        #remove the two batch list since the dataset has been modified
        try:
            del self.trainList
            del self.testList
        except AttributeError:
            pass





"""
    TEST SECTION
"""
def test():
    #datahandler object
    dh = dataHandler("hdf5")
    #fake data
    fake1=np.random.randint(255,size = [7056])
    fake2=np.random.randint(255,size = [4,4])
    fake3=np.random.randint(255,size = [512])
    for _ in range(60000):
        dh.addData(fake1,fake2,fake3)
    del fake1, fake2, fake3
    #save the elements in the buffer
    dh.saveData()



