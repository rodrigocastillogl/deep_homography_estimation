import os
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
    print(f'Saved as {data_directory}')

    
data_url = 'http://images.cocodataset.org/zips/train2017.zip'

root_path = './'
dataset_path = root_path + 'train2017'
zip_path = root_path + 'ms-coco.zip'

if os.path.exists(dataset_path):
    #   show a message and do nothing
    print( 'The dataset already exists at ' + dataset_path )

else:
    # download dataset zip file
    download_data(data_url, zip_path)
    
    # extract data from zip file
    print('Extracting MS-COCO 2017 dataset (it may take a while)...')
    with zipfile.ZipFile(zip_path, 'r') as archive:
        archive.extractall(root_path)
    print('Completed')

    # remove zip file
    os.remove(zip_path)
    print(zip_path + ' deleted.')