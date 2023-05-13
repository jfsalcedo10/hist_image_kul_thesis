import boto3
import os
from dotenv import load_dotenv


class AWSHandler:
    def __init__(self, aws_access_key_id, aws_secret_key, aws_region):

        # Creating the client instance
        self.aws_session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

        # Instatiate all clients needed
        self.s3_client = self.aws_session.client('s3')

    def list_folders(self, bucket_name, prefix='CAMELYON17/', delimiter='/', folders=False):
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
        response = self.s3_client.list_objects_v2(
            Bucket=bucket_name, Prefix=prefix, Delimiter=delimiter)
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

    def download_file(self, bucket_name, element_key, file_path, folder_path=''):
        final_path = file_path + folder_path + element_key.split('/')[-1]
        try:
            os.makedirs(file_path+folder_path)
        except OSError as e:
            if not os.path.isdir(file_path + folder_path):
                raise e

        self.s3_client.download_file(bucket_name, element_key, final_path)
        print(
            f'''The file {element_key.split('/')[-1]} has been downloaded in {final_path} succesfully.''')

    def download_list_file(self, list_files, bucket_name, file_path, folder_path='', all_files=True, starting_download_idx=0, files_to_download=10):
        if all_files:
            print(f'Downloading {len(list_files)} files: {list_files[:3]}...')
            for file in list_files:
                self.download_file(
                    bucket_name, file[0], file_path, folder_path)
        else:
            print(
                f'Downloading {files_to_download} files: {list_files[:3]}...')
            for file in list_files[starting_download_idx: starting_download_idx + files_to_download-1]:
                self.download_file(
                    bucket_name, file[0], file_path, folder_path)


if __name__ == '__main__':
    # Loading the AWS credentials
    load_dotenv('src/utils/aws_config.env')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_REGION = os.getenv('AWS_REGION')

    # Defining the bucket to extract info
    GOAL_BUCKET = 'camelyon-dataset'
    GOAL_DATASET = 'CAMELYON17/'
    GOAL_PATH = 'src/data/'
    STARTING_DOWNLOAD_IDX = 0
    FILES_TO_DOWNLOAD = 10

    aws_handler = AWSHandler(AWS_ACCESS_KEY_ID, AWS_SECRET_KEY, AWS_REGION)

    # Fetching the files outside a subfolders in the bucket
    folders_utils = aws_handler.list_folders(GOAL_BUCKET)
    del folders_utils[0]
    print(folders_utils)

    # Getting all the folders in the bucket
    folders_list = aws_handler.list_folders(GOAL_BUCKET, folders=True)
    print(folders_list)
