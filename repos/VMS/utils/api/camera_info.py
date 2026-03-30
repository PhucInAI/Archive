import  urllib.request
import  json
from    string              import  Template


URL_CAMERA                  =   Template("https://api.viact.ai/api/admin/monitor/$camera_id")


def get_camera_info(camera_id):
    """
        Get full camera info
    """

    url = URL_CAMERA.substitute(camera_id = camera_id)

    with urllib.request.urlopen(url) as url:
        data = json.loads(url.read().decode())

    return data