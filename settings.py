import os
import local_settings

DATA_DIR = local_settings.DATA_DIR

MODEL_DIR = os.path.join(DATA_DIR, 'models')

TRAIN_SIMPLIFIED_DIR = os.path.join(DATA_DIR, 'train_simplified')
SAMPLE_SUBMISSION = os.path.join(DATA_DIR, 'sample_submission.csv')
TEST_SIMPLIFIED = os.path.join(DATA_DIR, 'test_simplified.csv')

TEST_SIMPLIFIED_IMG_DIR = os.path.join(DATA_DIR, 'test-simplified-256')
