class CredentialsNotProvided(Exception):
    def __str__(self):
        return "Please provide credentials"


class IncorrectCredentials(Exception):
    def __str__(self):
        return "Please verify, that your credentials are right"


class IncorrectPolygon(Exception):
    def __str__(self):
        return "Provided polygon is invalid"


class IncorrectInput(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message
