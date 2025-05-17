from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np


class ImageData(BaseModel):
    name: str
    data: np.ndarray

    class Config:
        arbitrary_types_allowed = True


class ClusteringConfig(BaseModel):
    threshold: int = Field(default=5500, ge=1000, le=10000)
    max_images: int = Field(default=20, ge=2, le=50)
    max_file_size_mb: int = Field(default=10, ge=1, le=20)


class AppState(BaseModel):
    uploaded_images: list[ImageData] = []
    clusters: Optional[Dict[int, List[str]]] = None
    show_results: bool = False
    clustering_config: ClusteringConfig = ClusteringConfig()
    last_updated: Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
