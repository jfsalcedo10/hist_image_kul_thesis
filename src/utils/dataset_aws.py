import os
from utils.aws_handler import AWSHandler
from dotenv import load_dotenv

# Loading the AWS credentials
load_dotenv('../utils/aws_config.env')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY') 
AWS_REGION = os.getenv('AWS_REGION')

#Defining the bucket to extract info
GOAL_BUCKET = 'camelyon-dataset'
GOAL_DATASET = 'CAMELYON17/'
GOAL_PATH  = os.path.join('..','data')
STARTING_DOWNLOAD_IDX = 0
FILES_TO_DOWNLOAD = 10

aws_handler  = AWSHandler(AWS_ACCESS_KEY_ID, AWS_SECRET_KEY, AWS_REGION)
folders_utils = aws_handler.list_folders(GOAL_BUCKET)
del folders_utils[0]
folders_list = aws_handler.list_folders(GOAL_BUCKET, folders=True)
print(folders_list)

# Mapping files of each folder
annotations_list = aws_handler.list_folders(GOAL_BUCKET, folders_list[0])
evaluation_list = aws_handler.list_folders(GOAL_BUCKET, folders_list[1])
images_list = aws_handler.list_folders(GOAL_BUCKET,folders_list[2])
masks_list = aws_handler.list_folders(GOAL_BUCKET, folders_list[3])

# Downloading utils files from dataset
aws_handler.download_list_file(folders_utils, GOAL_BUCKET, GOAL_PATH)

# Downloading annotations from dataset
aws_handler.download_list_file(annotations_list, GOAL_BUCKET, GOAL_PATH, 'training/annotations/')

# # Downloading masks from dataset
aws_handler.download_list_file(masks_list, GOAL_BUCKET, GOAL_PATH, 'training/masks/')

# # Downloading evaulation scripts from bucket
aws_handler.download_list_file(evaluation_list, GOAL_BUCKET, GOAL_PATH, 'training/evaluation/')

# # Downloading images from dataset
aws_handler.download_list_file(images_list, GOAL_BUCKET, GOAL_PATH, 'training/images/', all_files= False)