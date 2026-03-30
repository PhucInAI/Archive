from utility.enums.server_types import ServerTypes
from utility.enums.system_modes import SystemModes
from utility.enums.system_status import SystemStatus


class LicenseFeatures:
    SERVER_TYPE = 'server_type'
    CUSTOMER_ID = 'customer_id'
    SECRET_KEY = 'secret_key'
    SYS_STATUS = 'sys_status'
    EXPIRED_DATE = 'expired_date'

    per_sys_keys = [SERVER_TYPE, CUSTOMER_ID, SECRET_KEY, SYS_STATUS, EXPIRED_DATE]

    MAX_DESKTOPS = 'max_desktops'  # per server_type
    MAX_SERVERS = 'max_servers'
    MAX_USERS = 'max_users'  # per server_type
    MAX_CAMERAS = 'max_cameras'  # per server_type
    SYS_MODE = 'sys_mode'  # per server_type

    per_app_keys = [MAX_DESKTOPS, MAX_SERVERS, MAX_USERS, MAX_CAMERAS, SYS_MODE]

    constraints = {
        SERVER_TYPE: {'type': 'str', 'values': ServerTypes.values},
        CUSTOMER_ID: {'type': 'str', 'values': None},
        SECRET_KEY: {'type': 'str', 'values': None},
        MAX_DESKTOPS: {'type': 'int', 'values': None},
        MAX_SERVERS: {'type': 'int', 'values': None},
        MAX_CAMERAS: {'type': 'int', 'values': None},
        MAX_USERS:  {'type': 'int', 'values': None},
        SYS_MODE: {'type': 'str', 'values': SystemModes.values},
        SYS_STATUS: {'type': 'str', 'values': SystemStatus.values},
        EXPIRED_DATE: {'type': 'date', 'values': None}
    }

    @staticmethod
    def normalize_license_value(license_feature, value):
        if LicenseFeatures.constraints[license_feature]['type'] == 'int':
            value = int(value)

        return value
