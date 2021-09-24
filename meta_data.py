

from tflite_support.metadata_writers import image_segmenter
from tflite_support.metadata_writers import writer_utils

ImageSegmenterWriter = image_segmenter.MetadataWriter
_MODEL_PATH = "/home/soojin/cgnet/cgnet_cityscapes.tflite"
# Task Library expects label files that are in the same format as the one below.
_LABEL_FILE = "/home/soojin/cgnet/label.txt"
_SAVE_TO_PATH = '/home/soojin/cgnet/cgnet_metadata.tflite'

_INPUT_NORM_STD = 127.5

# Create the metadata writer.
writer = ImageSegmenterWriter.create_for_inference(
    writer_utils.load_file(_MODEL_PATH), [_INPUT_NORM_MEAN], [_INPUT_NORM_STD],
    [_LABEL_FILE])

# Verify the metadata generated by metadata writer.
print(writer.get_metadata_json())

# Populate the metadata into the model.
writer_utils.save_file(writer.populate(), _SAVE_TO_PATH)