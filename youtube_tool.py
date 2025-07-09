import os
import re
import json
import asyncio
from datetime import datetime, timezone
from typing import List
from pydantic_ai import Tool, Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import httpx
from youtubesearchpython import VideosSearch as BaseVideosSearch
import yt_dlp
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from .prompts import get_chunk_summary_prompt, get_final_summary_prompt, get_cleaning_prompt
from .db_utils import get_collection
from .pydantic_models import ProductInput

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "meta-llama/Llama-2-7b-chat-hf"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = OpenAIModel("gpt-4o", provider=OpenAIProvider(api_key=OPENAI_API_KEY))
agent = Agent(OPENAI_MODEL)

CHUNK_SIZE = 3000

class CustomVideosSearch(BaseVideosSearch):
    def __init__(self, query, limit=10):
        super().__init__(query, limit=limit)
    
    def syncPostRequest(self):
        """Override to remove proxies argument."""
        try:
            response = httpx.post(
                url=self.url,
                headers=self.headers,
                json=self.data,
                timeout=10
            )
            response.raise_for_status()
            self.response = response.json()
        except Exception as e:
            print(f"Search request failed: {e}")
            self.response = {}

class YTDLogger:
    def __init__(self, log_file): 
        self.log_file = log_file
    
    def debug(self, msg): 
        self._write(msg)
    
    def warning(self, msg): 
        self._write(msg)
    
    def error(self, msg): 
        self._write(msg)
    
    def _write(self, msg): 
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

class VideoProcessor:
    def __init__(self, hf_token, hf_model):
        self.client = InferenceClient(model=hf_model, token=hf_token)

    @staticmethod
    def sanitize(name): 
        return re.sub(r'[\\/*?:"<>|]', "", name).strip()

    @staticmethod
    def extract_text_from_vtt(vtt):
        lines = vtt.splitlines()
        return '\n'.join(
            re.sub(r'<[^>]+>', '', line).strip()
            for line in lines
            if line.strip() and '-->' not in line and not line.startswith(('WEBVTT', 'Kind:', 'Language:')) and '[Music]' not in line
        )

    @staticmethod
    def remove_redundancy(text):
        lines = text.splitlines()
        seen = set()
        cleaned_lines = []
        previous_line = ""
        for line in lines:
            line = line.strip()
            if not line or line.lower() in {"uh", "um", "you know"}:
                continue
            if line != previous_line and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
            previous_line = line
        return '\n'.join(cleaned_lines)

    def clean_text(self, raw_text):
        prompt = get_cleaning_prompt(raw_text)
        try:
            return self.client.text_generation(
                prompt,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=False,
                stream=False
            ).strip()
        except Exception as e:
            print(f"HF cleaning error: {e}")
            return raw_text

    def get_video_urls(self, query, max_results=10):
        try:
            results = CustomVideosSearch(f"{query} demo OR review", limit=max_results).result()['result']
            return [r['link'] for r in results]
        except Exception as e:
            print(f"Failed to fetch video URLs: {e}")
            return []

    async def download_and_clean(self, video_url, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        try:
            with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
                info = ydl.extract_info(video_url, download=False)
                title = self.sanitize(info.get('title', 'video'))
                base_path = os.path.join(output_dir, title)
                vtt_file = f"{base_path}.en.vtt"
                json_file = f"{base_path}.json"
                log_file = f"{base_path}.log"

            if os.path.exists(json_file):
                return

            ydl_opts = {
                'quiet': True,
                'writesubtitles': True,
                'writeautomaticsub': True,
                'skip_download': True,
                'subtitleslangs': ['en'],
                'subtitlesformat': 'vtt',
                'outtmpl': base_path + '.%(ext)s',
                'logger': YTDLogger(log_file)
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])

            if os.path.exists(vtt_file):
                with open(vtt_file, 'r', encoding='utf-8') as f:
                    raw = f.read()
                loop = asyncio.get_running_loop()
                raw_text = self.extract_text_from_vtt(raw)
                precleaned = self.remove_redundancy(raw_text)
                cleaned = await loop.run_in_executor(None, self.clean_text, precleaned)
            else:
                cleaned = ""

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_title": title,
                    "video_url": video_url,
                    "transcript": cleaned
                }, f, indent=2, ensure_ascii=False)

            for f in [vtt_file, log_file]:
                if os.path.exists(f):
                    os.remove(f)

        except Exception as e:
            print(f"Download failed: {e}")

async def summarize_product_from_folder(product_name: str, transcript_dir: str):
    print(f"\U0001f9e0 Summarizing: {product_name}")
    all_texts = []
    video_objects = []

    for f in os.listdir(transcript_dir):
        if f.endswith(".json") and f != "combined_summary.json":
            with open(os.path.join(transcript_dir, f), 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if "transcript" in data:
                    all_texts.append(data["transcript"].strip())
                    video_objects.append({
                        "video_title": data.get("video_title", ""),
                        "video_url": data.get("video_url", ""),
                        "transcript": data.get("transcript", "")
                    })

    if not all_texts:
        return {"summary": "No usable transcripts."}

    combined_text = "\n\n".join(all_texts)
    chunks = [combined_text[i:i + CHUNK_SIZE] for i in range(0, len(combined_text), CHUNK_SIZE)]

    partial_summaries = []
    for i, chunk in enumerate(chunks):
        prompt = get_chunk_summary_prompt(product_name, chunk, i)
        try:
            result = await agent.run(prompt)
            summary = result.output if hasattr(result, "output") else str(result)
            partial_summaries.append(summary)
        except Exception as e:
            print(f"⚠️ Error summarizing chunk {i+1}: {e}")

    combined_summary_text = "\n\n".join(partial_summaries)
    final_prompt = get_final_summary_prompt(product_name, combined_summary_text)
    try:
        final_result = await agent.run(final_prompt)
        final_summary = final_result.output if hasattr(final_result, "output") else str(final_result)
    except Exception as e:
        print(f"⚠️ Final summarization failed: {e}")
        final_summary = combined_summary_text

    output_path = os.path.join(transcript_dir, "combined_summary.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "product_name": product_name,
            "summary": final_summary
        }, f, indent=2, ensure_ascii=False)

    collection = get_collection("company", "company_data")
    product_key = product_name.strip().lower().replace(" ", "_")
    company_name = product_name.split()[0].strip().lower()

    collection.update_one(
        {"company": company_name},
        {"$set": {
            f"youtube.{product_key}": {
                "videos": video_objects,
                "summary": final_summary,
                "scraped_at": datetime.now(timezone.utc).isoformat()
            }
        }},
        upsert=True
    )

    print(f"✅ Saved summary to MongoDB under youtube.{product_key}")
    return {"summary": final_summary}

async def _generate_product_summary_internal(product_name: str):
    processor = VideoProcessor(HF_TOKEN, HF_MODEL)
    base_dir = os.path.join("outputs", f"{VideoProcessor.sanitize(product_name)}_captions")
    os.makedirs(base_dir, exist_ok=True)

    urls = processor.get_video_urls(product_name, max_results=10)
    await asyncio.gather(*(processor.download_and_clean(url, base_dir) for url in urls))

    return await summarize_product_from_folder(product_name, base_dir)

@Tool
async def generate_product_summary(input: ProductInput):
    """Search YouTube for product demo/review videos, extract transcripts, and summarize key insights."""
    return await _generate_product_summary_internal(input.product_name)