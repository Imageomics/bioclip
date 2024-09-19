import dataclasses
import logging
import re


logger = logging.getLogger()

# ToL ID generated from uuid.uuid4(), uuid regex from https://stackoverflow.com/a/6640851
tol_filename_pattern = re.compile("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}.*jpg$")


@dataclasses.dataclass(frozen=True)
class ImageFilename:
    """
    Represents a filename like <uuid.uuid4()>.jpg, returns zero for the content and page IDs
    """

    tol_id: str
    ext: str
    raw: str

    @classmethod
    def from_filename(cls, filename):
        match = tol_filename_pattern.match(filename)
        if not match:
            raise ValueError(filename)
        return cls(filename.split(".")[0], "jpg", filename)
