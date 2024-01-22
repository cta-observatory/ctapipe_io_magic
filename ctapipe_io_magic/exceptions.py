class MissingInputFilesError(Exception):
    """
    Exception raised when there are no input files.
    """

    pass


class FailedFileCheckError(Exception):
    """
    Exception raised when the files check fails.
    """

    pass


class MissingDriveReportError(Exception):
    """
    Exception raised when a subrun does not have drive reports.
    """

    pass
