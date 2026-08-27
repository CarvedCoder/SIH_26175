from pydantic import BaseModel

class ImageMetadata(BaseModel):
    filename: str
    file_type: str
    width: int | None = None
    height: int | None = None 
    Channels: int | None = None 
    has_geospatial_metadata: bool = False
    crs: str | None = None