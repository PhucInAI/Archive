import json

from flask import jsonify
from utility.exceptions.security_exceptions import LicenseNotFoundException, InvalidLicenseException, \
    LimitedLicenseException, KickOutException
from utility.exceptions.user_exception import UserException


class JsonResult:
    def __init__(self, success=None, data=None, details=None):
        self.success = success
        self.data = data
        self.details = details

    def json(self):
        return jsonify(self.to_dict())

    def json_error(self, exception):
        return jsonify(self.error(exception))

    def error(self, exception):
        self.success = False
        if type(exception) is UserException:
            self.data = exception.args[0]
            self.details = exception.errors
        elif type(exception) is LicenseNotFoundException:
            self.data = 'LICENSE_NOT_FOUND'
            self.details = exception.errors
        elif type(exception) is InvalidLicenseException:
            self.data = 'INVALID_LICENSE'
            self.details = exception.errors
        elif type(exception) is LimitedLicenseException:
            self.data = 'LIMITED_LICENSE'
            self.details = exception.errors
        elif type(exception) is KickOutException:
            self.data = 'CLIENT_KICKED_OUT'
            self.details = exception.errors
        else:
            self.data = 'SYSTEM_ERROR'
            self.details = 'Unexpected error has happened. You may try again or inform the administrator'

        return self.to_dict()

    def to_dict(self):
        result = {
            'success': self.success,
            'data': self.data,
            'details': self.details,
            'next_page_token': None
        }

        if hasattr(self, 'next_page_token'):
            result['next_page_token'] = self.next_page_token
        if hasattr(self, 'total_pages'):
            result['total_pages'] = self.total_pages
        if hasattr(self, 'total_items'):
            result['total_items'] = self.total_items

        return result

    def json_paginate(self, next_page_token, total_pages=None, total_items=None):
        return jsonify(self.dict_paginate(next_page_token, total_pages=total_pages, total_items=total_items))

    def dict_paginate(self, next_page_token, total_pages=None, total_items=None):
        self.next_page_token = next_page_token
        if total_pages is not None:
            self.total_pages = total_pages
        if total_items is not None:
            self.total_items = total_items

        return self.to_dict()

    @staticmethod
    def load(json_string):
        obj = json.loads(json_string)
        result = JsonResult()
        if 'success' in obj:
            result.success = obj['success']

        if 'data' in obj:
            result.data = obj['data']

        if 'details' in obj:
            result.details = obj['details']

        if 'total_pages' in obj:
            result.total_pages = obj['total_pages']

        if 'total' in obj:
            result.total = obj['total']

        return result
