import asyncio
from youtube_scraper_tool import generate_product_summary
from pydantic_models import ProductInput

if __name__ == "__main__":
    product = ProductInput(product_name="iPhone 15")  # 🔁 Replace with any product
    asyncio.run(generate_product_summary(product))
