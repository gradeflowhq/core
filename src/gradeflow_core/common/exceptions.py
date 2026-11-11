class GradeFlowError(Exception):
    pass


class EditingError(GradeFlowError):
    """A request to edit a marking names a path or method that doesn't fit where it is."""
