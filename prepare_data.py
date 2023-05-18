import os
from glob import glob
#from shutil import rmtree
import errno
import wget
import zipfile

def download_data( data_url = None, data_directory = None ):
    """
    Download dataset.
    Input
    -----
        * data_url: url to download dataset.
        * data_directory : diretory to save downloaded file.
    Output
    ------
        *  None
    """

    assert data_url is not None
    assert data_directory is not None

    print(f'Downloading dataset (it may take a while)...')
    response = wget.download(data_url, data_directory)
    print(f'Save as {data_directory}')


def split_data( data_path = None, ratio = 0.8 ):
    pass
    


if __name__ == '__main__':

    data_url = 'http://images.cocodataset.org/zips/train2017.zip'


    dataset_directory = 'data'
    train_directory = dataset_directory + '/train'
    test_directory  = dataset_directory + '/test'

    # create dataset directory
    try:
        os.makedirs(dataset_directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if os.path.exists(train_directory) and os.path.exists(test_directory):
        # if train and test directories already exists:
        #   show a message and do nothing
        print( 'The dataset already exists at ' + train_directory +
               'and' + test_directory  )
    else:

        # download dataset zip file
        zip_directory = dataset_directory + "/ms-coco.zip"
        download_data(data_url, dataset_directory)

        # extract fata from zip file
        print('Extracting MS-COCO 2017 dataset ...')
        with zipfile.ZipFile(zip_directory, 'r') as archive:
            archive.extractall(dataset_directory)
        # remove zip file
        os.remove(zip_directory)


        # create train and test directories
        try:
            os.makedirs(train_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
        try:
            os.makedirs(test_directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise