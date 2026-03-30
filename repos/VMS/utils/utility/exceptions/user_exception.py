class UserException(Exception):
    def __init__(self, error_code, error_message=''):
        super(UserException, self).__init__(error_code)
        self.errors = error_message
        self.error_code = error_code
