from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

def load_labelmap(path):
    with open(path, 'r') as fid:
        label_map_string = fid.read()
        label_map = string_int_label_map_pb2.StringIntLabelMap()
        try:
            text_format.Merge(label_map_string, label_map)
        except text_format.ParseError:
            label_map.ParseFromString(label_map_string)
    _validate_label_map(label_map)
    return label_map

def _validate_label_map(label_map):
    # Add your validation logic here
    if not label_map.item:
        raise ValueError('Label map is empty.')
    for item in label_map.item:
        if item.id < 1:
            raise ValueError('Label map item ID must be greater than or equal to 1.')
        if not item.name:
            raise ValueError('Label map item name is missing.')
    return True
