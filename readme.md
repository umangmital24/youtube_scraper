# YouTube Scraper Tool

This tool searches YouTube for a given product's demos/reviews, downloads English transcripts, cleans and summarizes them using LLaMA and GPT, and stores everything in MongoDB.

### Structure

- `db_utils.py` - Handles MongoDB connection
- `pydantic_models.py` - Defines input schema
- `youtube_scraper_tool.py` - Full pipeline: search → transcript → clean → summarize → store
- `ss/` - Place for screenshots or logs

### Setup

1. Create `.env` with:
