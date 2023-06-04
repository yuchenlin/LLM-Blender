import hashlib
import argparse 

def generate_hash_code(text):
    # Convert the text to bytes and create a hash object
    hash_object = hashlib.sha256(text.encode())

    # Get the hexadecimal representation of the hash code
    hex_code = hash_object.hexdigest()

    # Return the first 16 digits of the hexadecimal code
    return hex_code[:16]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def empty2None(x):
    if x == '':
        return None
    else:
        return x

def empty2zero(x):
    if x == '':
        return 0
    elif isinstance(x, int):
        return x
    elif isinstance(x, str):
        return int(x)
    else:
        raise argparse.ArgumentTypeError('Integer value expected.')
