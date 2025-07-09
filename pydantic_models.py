from pydantic import BaseModel

class ProductInput(BaseModel):
    product_name: str
