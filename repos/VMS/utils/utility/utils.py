import collections
import hashlib
import io
import json
import os
import random
import re
import shutil
import socket
import string
import subprocess
import tempfile
import time
import uuid
import zipfile
from collections import namedtuple
from datetime import datetime, timedelta
from enum import Enum
from subprocess import Popen, call
from typing import Optional, NoReturn, Sized

import cv2
import numpy as np
import pytz
import unidecode

from utility.enums.error_codes import ErrorCode
from utility.exceptions.user_exception import UserException


def normalize_context_root(ctx):
    # Make sure context root always be: /<something or empty>
    return '/' + ('' if ctx is None else ctx).strip().strip('/')


def p_kill(process): return process.kill()


class Utils:
    def __init__(self):
        self.__timestamp = int(datetime.timestamp(datetime.now()))

    @staticmethod
    def is_null_or_empty(obj):
        return obj is None or obj == '' or (isinstance(obj, str) and obj.strip() == '')

    @staticmethod
    def empty_to_none(obj) -> Optional[str]:
        if obj is None:
            return None
        s = str(obj).strip()
        if s == '':
            return None
        return s

    @staticmethod
    def strip_redundant_spaces(s: Optional[str]) -> Optional[str]:
        new_str = Utils.empty_to_none(s)
        if new_str is None:
            return None
        return re.sub(r'\s+', ' ', s.strip())

    @staticmethod
    def to_nullable_bool(obj) -> bool:
        return None if Utils.is_null_or_empty(obj) else Utils.to_boolean(obj)

    @staticmethod
    def to_boolean(obj) -> bool:
        if Utils.is_null_or_empty(str(obj)):
            return False
        if isinstance(obj, bool):
            return obj

        return str(obj).lower() == 'true' or obj == 1 or str(obj) == '1'

    @staticmethod
    def to_int(obj) -> Optional[int]:
        if Utils.is_null_or_empty(obj):
            return None
        return int(obj)

    @staticmethod
    def check_null_args(args, names):
        for index, arg in enumerate(args):
            if Utils.is_null_or_empty(arg):
                raise UserException('ARGURMENT_REQUIRED', names[index] + ' is required.')

    @staticmethod
    def generate_id(size=6):
        return ''.join(random.choice(string.ascii_uppercase) for _ in range(size))

    @staticmethod
    def id_from_mac_address(mac):
        return mac.replace(':', '').replace('-', '').replace('_', '').strip().lower()

    @staticmethod
    def generate_time_id(dt=None):
        if dt is not None:
            return dt.strftime('%Y%m%d-%H%M%S-') + str(uuid.uuid4())

        return datetime.now().strftime('%Y%m%d-%H%M%S-') + str(uuid.uuid4())

    @staticmethod
    def get_date_from_time_id(id):
        if Utils.is_null_or_empty(id):
            return None

        parts = id.split('-')
        if len(parts) > 1:
            return datetime.strptime(parts[0] + '-' + parts[1], '%Y%m%d-%H%M%S')

        return None

    @staticmethod
    def json_collection(collection):
        json = []
        for obj in collection:
            json.append(obj.json())

        return json

    @staticmethod
    def json_collection_with(collection, func_name):
        json = []
        for obj in collection:
            func = getattr(obj, func_name)
            json.append(func())

        return json

    @staticmethod
    def json_for_map_collection(collection):
        json = []
        for obj in collection:
            json.append(obj.json_for_map())

        return json

    @staticmethod
    def get_current_datetime():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def tostring(dt):
        if dt is None or dt == '':
            return None

        return dt.strftime('%Y-%m-%d %H:%M:%S')

    @staticmethod
    def ONVIF_scaner(filename):
        return call(["onvif_scan", filename])

    @staticmethod
    def get_local_ipv4_address():
        return socket.gethostbyname(socket.gethostname())

    @staticmethod
    def prRed(prt):
        print("\033[91m {}\033[00m".format(prt))

    @staticmethod
    def np_array_to_bits(items):
        arr = np.asarray(items)
        mem = io.BytesIO()
        np.save(mem, arr)
        mem.seek(0)
        bits = mem.getvalue()
        mem.close()
        return bits

    @staticmethod
    def np_item_to_bits(item):
        mem = io.BytesIO()
        np.save(mem, item)
        mem.seek(0)
        bits = mem.getvalue()
        mem.close()
        return bits

    @staticmethod
    def bits_to_np_array(bits, element_len):
        mem = io.BytesIO(bits)
        arr = np.load(mem)
        mem.close()
        X = []
        for i, item in enumerate(arr):
            Utils.extract_np_element(X, item, element_len)

        return X

    @staticmethod
    def extract_np_element(result, item, element_len):
        if len(item) == element_len:
            result.append(item)
        else:
            for i, x in enumerate(item):
                Utils.extract_np_element(result, x, element_len)

    @staticmethod
    def bits_to_np_item(bits):
        mem = io.BytesIO(bits)
        return np.load(mem)

    @staticmethod
    def resize_image(image, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation=inter)
        return resized

    @staticmethod
    def caused_by_lost_connection(self, e):
        message = str(e).lower()
        for s in ["server has gone away",
                  "no connection to the server",
                  "lost connection",
                  "is dead or not enabled",
                  "error while sending",
                  "decryption failed or bad record mac",
                  "server closed the connection unexpectedly",
                  "ssl connection has been closed unexpectedly",
                  "error writing data to the connection",
                  "connection timed out",
                  "resource deadlock avoided"]:
            if s in message:
                return True

        return False

    @staticmethod
    def json2obj(data):
        return json.loads(data, object_hook=Utils._json_object_hook)

    @staticmethod
    def _json_object_hook(d):
        return namedtuple('X', d.keys())(*d.values())

    @staticmethod
    def get_disk_info(mount_folder, timeout):
        result = {'percent': -1, 'size': -1, 'used': -1, 'available': -1}
        start = time.time()
        p = Popen(["df", "-BG", mount_folder], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        while time.time() - start < timeout:
            if p.poll() is not None:
                stdout, stderr = p.communicate()
                items = stdout.decode('ascii').splitlines()
                if len(items) == 0:
                    return result

                result['percent'] = float(items[1].split()[4].split("%")[0])
                result['size'] = float(items[1].split()[1].split("G")[0])
                result['used'] = float(items[1].split()[2].split("G")[0])
                result['available'] = float(items[1].split()[3].split("G")[0])
                return result

            time.sleep(0.1)

        p.kill()
        p.terminate()
        p.wait()
        if p.stdout:
            p.stdout.close()

        if p.stdin:
            p.stdin.close()

        if p.stderr:
            p.stderr.close()

        return result

    def timestamp(self):
        self.__timestamp += 1
        return self.__timestamp

    @staticmethod
    def compare_time(t1, t2):
        time1 = datetime.strptime(t1, '%H:%M:%S')
        delta1 = timedelta(hours=time1.hour, minutes=time1.minute, seconds=time1.second)
        time2 = datetime.strptime(t2, '%H:%M:%S')
        delta2 = timedelta(hours=time2.hour, minutes=time2.minute, seconds=time2.second)

        return delta1 > delta2

    @staticmethod
    def compare_primitive(o1, o2) -> int:
        if o1 == o2:
            return 0
        if o1 is None:
            return -1
        if o2 is None:
            return 1
        return 1 if o1 > o2 else -1

    @staticmethod
    def uppercase(s: Optional[str]) -> Optional[str]:
        return s if s is None else s.upper()

    @staticmethod
    def get_system_info():
        import platform
        info = {}
        info['Architecture'] = platform.architecture()[0]
        info['Machine'] = platform.machine()
        info['Node'] = platform.node()
        # get processor info
        with open("/proc/cpuinfo", "r") as f:
            cpu_info = f.readlines()
        info['Processor'] = [x.strip().split(":")[1] for x in info if "model name" in x]
        info['System'] = platform.system()
        dist = platform.dist()
        dist = " ".join(x for x in dist)
        info['Distribution'] = dist
        with open("/proc/loadavg", "r") as f:
            load = f.read().strip()
        info['Load'] = load
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info['Mem'] = 'Total=' + str(lines[0].strip()) + 'Free=' + str(lines[1].strip())
        return info

    @staticmethod
    def zip_dir(path, output_file):
        zipf = zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(path):
            for file in files:
                zipf.write(os.path.join(root, file))

        zipf.close()

    @staticmethod
    def extract_zip(zip_file_path, output_dir):
        with zipfile.ZipFile(zip_file_path) as zip_file:
            for member in zip_file.namelist():
                filename = os.path.basename(member)
                if not filename:  # skip directories
                    continue

                # copy file (taken from zipfile's extract)
                source = zip_file.open(member)
                target = open(os.path.join(output_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)

    @staticmethod
    def save_to_temp_file(bytes):
        fd, tmp = tempfile.mkstemp()
        with open(tmp, 'wb') as f:
            f.write(bytes)
            f.close()

        return tmp

    @staticmethod
    def get_date_range(start_at, end_at, time_id):
        search_start_on = Utils.get_date_from_time_id(time_id)
        if search_start_on is not None:
            search_start_on = search_start_on.date()
        else:
            search_start_on = start_at.date()

        all_dates = []
        while search_start_on <= end_at.date():
            all_dates.append(search_start_on)
            search_start_on += timedelta(days=1)

        return all_dates

    @staticmethod
    def to_dict(obj, enum_to_str=False):
        """
        Recursively turn the object graph into dict.
        WARNING: this implementation may not work in all cases, always check its return
        """
        if obj is None:
            return None

        if isinstance(obj, str):
            return obj
        elif isinstance(obj, Enum):
            if enum_to_str:
                return obj.__str__()
            return obj
        elif isinstance(obj, dict):
            return dict((key, Utils.to_dict(val, enum_to_str=enum_to_str)) for key, val in obj.items())
        elif isinstance(obj, collections.Iterable):
            return [Utils.to_dict(val, enum_to_str=enum_to_str) for val in obj]
        elif hasattr(obj, 'as_dict_overridden') and callable(getattr(obj, 'as_dict_overridden')):
            return Utils.to_dict(obj.as_dict_overridden(), enum_to_str=enum_to_str)
        elif hasattr(obj, '__dict__'):
            return Utils.to_dict(vars(obj), enum_to_str=enum_to_str)
        elif hasattr(obj, '__slots__'):
            return Utils.to_dict(dict((name, getattr(obj, name)) for name in getattr(obj, '__slots__')),
                                 enum_to_str=enum_to_str)
        return obj

    @staticmethod
    def from_epoch_utc(seconds_utc: int) -> Optional[datetime]:
        if seconds_utc is None or seconds_utc == 0:
            return None
        naive_datetime = datetime.utcfromtimestamp(seconds_utc)
        utc_dt: datetime = pytz.utc.localize(naive_datetime)
        return utc_dt

    @staticmethod
    def enforce_int_range(val, field_name: str,
                          min_value: Optional[int] = None, max_value: Optional[int] = None) -> Optional[int]:
        if Utils.is_null_or_empty(val):
            return None
        int_val = int(val)
        if min_value is not None and int_val < min_value:
            raise UserException(ErrorCode.INVALID_DATA, f'{field_name} must not be smaller than {min_value}.'
                                                        f' But was {val}')
        if max_value is not None and int_val > max_value:
            raise UserException(ErrorCode.INVALID_DATA, f'{field_name} must not be greater than {max_value}.'
                                                        f' But was {val}')
        return int(int_val)

    @staticmethod
    def exclude_none_attributes(a_dict: Optional[dict]) -> Optional[dict]:
        if a_dict is None:
            return None
        return {k: v for k, v in a_dict.items() if v is not None}

    @staticmethod
    def extract_modifications(new_state: dict, old_state: dict) -> dict:
        """:returns items from new_state that doesn't exist in old_state or its value is different"""
        modified_state = {}
        for att in new_state:
            if att not in old_state or old_state[att] != new_state[att]:
                modified_state[att] = new_state[att]
        return modified_state

    @staticmethod
    def remove_accents(txt: Optional[str]) -> Optional[str]:
        if txt is None:
            return txt
        return unidecode.unidecode(txt)

    @staticmethod
    def mandatory(value, info_name) -> NoReturn:
        """
            :raise UserException if the provided "value" is empty or None. Otherwise, nothing will happen
            :param value value to check
            :param info_name will be used to build the exception message
        """
        if Utils.is_null_or_empty(value) or (isinstance(value, Sized) and len(value) == 0):
            raise UserException(ErrorCode.INVALID_DATA, f'{info_name} is required but not provided')

    @staticmethod
    def get_unique_id(s):
        return f'{int(hashlib.md5(str.encode(s)).hexdigest(), 16)}'[0:16]
