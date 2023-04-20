import boto3 
import os
from dotenv import load_dotenv

# Loading the AWS credentials
load_dotenv('src/utils/aws_config.env')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY') 
AWS_REGION = os.getenv('AWS_REGION')

#Defining the bucket to extract info
GOAL_BUCKET = 'camelyon-dataset'
GOAL_DATASET = 'CAMELYON17/'
GOAL_PATH  = 'src/data/'
STARTING_DOWNLOAD_IDX = 0
FILES_TO_DOWNLOAD = 10

def list_folders(s3_client, bucket_name, prefix = 'CAMELYON17/', delimiter = '/', folders = False):
    """Gets all objects in the given S3 bucket. If folders is True, gets all 
    the common prefixes of the bucket. 

    Args:
        s3_client (_type_): _description_
        bucket_name (_type_): _description_
        prefix (str, optional): _description_. Defaults to 'CAMELYON17/'.
        delimiter (str, optional): _description_. Defaults to '/'.
        folders (bool, optional): _description_. Defaults to False.

    Returns:
        elements_list: Array of tuples containing the key (and size in MB if folders=False) of 
        each element in the bucket.
    """
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)
    elements_list = []
    if folders:
        for prefix in response.get('CommonPrefixes', []):
            prefix_name = prefix.get('Prefix')
            elements_list.append(prefix_name)
    else:
        for element in response.get('Contents', []):
            element_key = element.get('Key')
            element_size = element.get('Size')/1024/1024
            elements_list.append([element_key, element_size])
    return elements_list

# Creating the client instance
aws_session = boto3.Session(
    aws_access_key_id = AWS_ACCESS_KEY_ID,
    aws_secret_access_key = AWS_SECRET_KEY,
    region_name = AWS_REGION
)

def download_file(s3_client,bucket_name, element_key, file_path, folder_path=''):
    final_path = file_path + folder_path + element_key.split('/')[-1]
    try:
        os.makedirs(file_path+folder_path)
    except OSError as e:
        if not os.path.isdir(file_path+ folder_path):
            raise e

    s3_client.download_file(bucket_name, element_key, final_path)
    print(f'''The file {element_key.split('/')[-1]} has been downloaded in {final_path} succesfully.''')

def download_list_file(list_files,s3_client,bucket_name, file_path, folder_path='', all_files=True):    
    if all_files:
        print(f'Downloading {len(list_files)} files: {list_files[:3]}...')
        for file in list_files:
            download_file(s3_client, bucket_name, file[0], file_path, folder_path)
    else:
        print(f'Downloading {FILES_TO_DOWNLOAD} files: {list_files[:3]}...')
        for file in list_files[STARTING_DOWNLOAD_IDX: STARTING_DOWNLOAD_IDX +FILES_TO_DOWNLOAD-1]:
            download_file(s3_client, bucket_name, file[0], file_path, folder_path)

if __name__ == '__main__':
    # Fetching the desired dataset
    s3 = aws_session.client('s3')
    folders_utils = list_folders(s3,GOAL_BUCKET)
    del folders_utils[0]

    # Getting all the folders in the bucket
    folders_list = list_folders(s3,GOAL_BUCKET, folders=True)

    # Mapping files of each folder
    annotations_list = list_folders(s3,GOAL_BUCKET, folders_list[0])
    evaluation_list = list_folders(s3,GOAL_BUCKET, folders_list[1])
    images_list = list_folders(s3,GOAL_BUCKET,folders_list[2])
    masks_list = list_folders(s3, GOAL_BUCKET, folders_list[3])

    print(masks_list)
    # Downloading utils files from dataset
    # download_list_file(folders_utils, s3, GOAL_BUCKET, GOAL_PATH)

    # Downloading annotations from dataset
    download_list_file(annotations_list, s3, GOAL_BUCKET, GOAL_PATH, 'training/annotations/')

    # # Downloading masks from dataset
    # download_list_file(masks_list, s3, GOAL_BUCKET, GOAL_PATH, 'training/masks/')

    # # Downloading evaulation scripts from bucket
    # download_list_file(evaluation_list, s3, GOAL_BUCKET, GOAL_PATH, 'training/evaluation/')

    # # Downloading images from dataset
    # download_list_file(images_list, s3, GOAL_BUCKET, GOAL_PATH, 'training/images/', all_files= False)