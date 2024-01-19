class MissingInputFilesError(Exception):
    """
    Exception raised when there are no input files.
    """

    def __init__(self, message):
        self.message = message


class FailedFileCheckError(Exception):
    """
    Exception raised when the files check fails.
    """

    def __init__(self, message):
        self.message = message


class MissingDriveReportError(Exception):
    """
    Exception raised when a subrun does not have drive reports.
    """

    def __init__(self, message):
        self.message = message
