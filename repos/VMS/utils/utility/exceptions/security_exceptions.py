class KickOutException(Exception):
    def __init__(self, error_code, error_message=''):
        super(KickOutException, self).__init__(error_code)
        self.errors = error_message


class LicenseNotFoundException(Exception):
    def __init__(self, error_code, error_message=''):
        super(LicenseNotFoundException, self).__init__(error_code)
        self.errors = error_message


class InvalidLicenseException(Exception):
    def __init__(self, error_code, error_message=''):
        super(InvalidLicenseException, self).__init__(error_code)
        self.errors = error_message


class LimitedLicenseException(Exception):
    def __init__(self, error_code, error_message=''):
        super(LimitedLicenseException, self).__init__(error_code)
        self.errors = error_message
