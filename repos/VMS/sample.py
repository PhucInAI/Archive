import boto3
from botocore.config import Config as Botoconfig
import requests
import datetime
import os
import urllib.parse
import json
import queue
import threading
from string import Template
import time
import sys

EVENT_OBJECT_KEY_TEMPLATE = Template("event/production/$device_id/$monitor_id/$date_ymd/$basename")
SNAPSHOT_OBJECT_KEY_TEMPLATE = Template("thumbnail/production1/$monitor_id.jpg")
PUBLIC_URL_TEMPLATE = Template("https://$bucket.s3.$region.amazonaws.com/$object_key")


# or using a python package "backoff" which enables a more fine grained control on retry behaviour
def simple_retry(interval=30, max_retries=None, exceptions = Exception):
    def inner(func):
        def wrapper(*args, **kwargs):
            attempts = 0
            while True:
                try:
                    res = func(*args, **kwargs)
                    return res
                except exceptions as e:
                    print("Error function {}: {}".format(func.__name__, e))
                    attempts +=1
                    if max_retries is not None and attempts >= max_retries:
                        raise e
                    time.sleep(interval)

        return wrapper
    return inner


class S3Client:
    def __init__(self, 
                 region="ap-east-1",
                 host=None,
                 use_ssl=True, 
                 access_key_id=None,
                 secret_access_key=None,
                 connect_timeout=8):
        extra_args = {}

        if access_key_id is not None:
            extra_args["aws_access_key_id"] = access_key_id
        if secret_access_key is not None:
            extra_args["aws_secret_access_key"] = secret_access_key
        if host is not None:
            extra_args["endpoint_url"] = host
        if use_ssl is not None:
            extra_args["use_ssl"] = use_ssl
        
        config = Botoconfig(connect_timeout = connect_timeout, 
                            max_pool_connections=30, 
                            retries={"max_attempts":1})
        self.client = boto3.client("s3", region_name = region, config=config, **extra_args)
        self.region = region

    @simple_retry()
    def upload_file(self, path, bucket, key, public_read=True):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} does not exist")

        extra_args = {}
        if public_read:
            extra_args["ACL"] = "public-read"

        self.client.upload_file(path, bucket, key, ExtraArgs = extra_args)

class ViactaiClient:
    def __init__(self, url, timeout=30):
        url = url+"/" if not url.endswith("/") else url
        self.url = url
        self.timeout = 30
        self.headers = {}

    @simple_retry()
    def send_event(self, event, monitor_id, video_url=None, image_url=None):
        payload = {
            "alert": "Y",
            "result": "Y",
            "engine": event,
            "monitor_id": monitor_id
        }
        if video_url is not None:
            payload["video_url"] = video_url
        if image_url is not None:
            payload["image_url"] = image_url

        resp = requests.post(
                urllib.parse.urljoin(self.url, "detection/incoming"),
                headers=self.headers,
                json=payload,
                timeout=self.timeout,
        )
        if int(resp.status_code //100) ==2:
            return {"status":"success","data": json.loads(resp.content.decode())}
        else:
            return {"status":"fail","data": resp.content}
        
    def update_monitor_snapshot(self, monitor_id, image_url):
        resp = requests.put(
                urllib.parse.urljoin(self.url, f"monitor/{monitor_id}"),
                headers=self.headers,
                json={"snapshot": image_url},
                timeout=15,
        )
        print(resp)
        sys.exit()


class EventData:
    def __init__(self,event, monitor_id, image_path, video_path, destination_s3_bucket, device_id):
        self.event = event
        self.monitor_id =monitor_id
        self.image_path = image_path
        self.video_path = video_path
        self.destination_s3_bucket = destination_s3_bucket
        self.device_id = device_id

class EventManager:
    def __init__(self, num_workers=3, viact_api_url="https://api.viact.ai/api/admin"):
        self.viactai_cli = ViactaiClient(viact_api_url)
        self.s3_cli = S3Client()
        self.tasks = queue.Queue()

        self.threads = []
        for _ in range(num_workers):
            self.threads.append(threading.Thread(target=self.process, args=()))
        for t in self.threads:
            t.start()

    def add_event(self, event, monitor_id, image_path, video_path, destination_s3_bucket, device_id="device"):
        """
        Parameters
        ----------
        event : str
            ID of engine, e.g. danger-zone
        monitor_id : str
        image_path : str
            local path of the image. Must be unique otherwise new file will overwrite the old file in the bucket since no versioning is enabled
        video_oath  : str
            local path of the video. Must be unique otherwise new file will overwrite the old file in the bucket since no versioning is enabled
        destination_s3_bucket : str
            name of the s3 bucket storing the alerts images and alert videos
        device_id : str 
        """
        self.tasks.put(EventData(event, monitor_id, image_path, video_path, destination_s3_bucket, device_id))

    def process(self):
        while True:
            event_data = self.tasks.get()
            event = event_data.event
            monitor_id = event_data.monitor_id
            image_path = event_data.image_path
            video_path = event_data.video_path
            device_id = event_data.device_id
            bucket = event_data.destination_s3_bucket

            date_ymd = str(datetime.datetime.now())[:10]

            image_basename = os.path.basename(image_path)
            video_basename = os.path.basename(video_path)

            image_object_key = EVENT_OBJECT_KEY_TEMPLATE.substitute(device_id=device_id, monitor_id=monitor_id, basename = image_basename, date_ymd = date_ymd)
            video_object_key = EVENT_OBJECT_KEY_TEMPLATE.substitute(device_id=device_id, monitor_id=monitor_id, basename = video_basename, date_ymd = date_ymd)

            image_url = PUBLIC_URL_TEMPLATE.substitute(region=self.s3_cli.region, bucket = bucket, object_key = image_object_key)
            video_url = PUBLIC_URL_TEMPLATE.substitute(region=self.s3_cli.region, bucket = bucket, object_key = video_object_key)

            self.s3_cli.upload_file(image_path, bucket, image_object_key)
            self.s3_cli.upload_file(video_path, bucket, video_object_key)

            # This must be done after successfully uploading files to s3.
            self.viactai_cli.send_event(event, monitor_id, video_url = video_url, image_url=image_url)

    def update_snapshot(self, monitor_id, image_path, bucket, viact_api_url="https://api.viact.ai/api/admin"):
        viactai_cli.update_monitor_snapshot(monitor_id=monitor_id, image_url=snapshot_url)
        viactai_cli = ViactaiClient(viact_api_url)
        s3_cli = S3Client()

        snapshot_object_key = SNAPSHOT_OBJECT_KEY_TEMPLATE.substitute(monitor_id = monitor_id)
        s3_cli.upload_file(image_path, bucket, snapshot_object_key)
        snapshot_url = PUBLIC_URL_TEMPLATE.substitute(bucket=bucket, region=s3_cli.region, object_key = snapshot_object_key)
        viactai_cli.update_monitor_snapshot(monitor_id=monitor_id, image_url=snapshot_url)
              

if __name__ == "__main__":
    # ev_manager = EventManager()
    # for _ in range(10):
    #     ev_manager.add_event("danger-zone", "omni-cam00",
    #                     "H:/aaa_v02-cam00_aaa_v02_48-b0-2d-15-e2-3f_1682487539.6795883.jpg",
    #                     "H:/aaa_v02-cam00_aaa_v02_48-b0-2d-15-e2-3f_1682487539.6795883.mp4", 
    #                     "viact-deployment-others")
    
    # ev_manager.update_snapshot("omni-cam00", "H:/aaa_v02-cam00_aaa_v02_48-b0-2d-15-e2-3f_1682487539.6795883.jpg",bucket="viact-deployment-others")
    # time.sleep(30)
    # print('here')
    viactai_cli = ViactaiClient(url="https://api.viact.ai/api/admin")
    resq    =   viactai_cli.update_monitor_snapshot("omni-cam00")
    url     =   'https://api.viact.ai/api/admin/monitor/omni-cam00'
    data    =   {'name': 'John', 'age': 30}
    json_data = json.dumps(data)

    response = requests.post(url, data=json_data)

    print(response.status_code)
    print(response.content.decode())