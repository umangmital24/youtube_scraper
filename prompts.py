def get_chunk_summary_prompt(product_name: str, chunk_text: str, chunk_index: int) -> str:
    return (
        f"Chunk {chunk_index + 1}:\n"
        f"Summarize this transcript chunk focusing only on the product '{product_name}', "
        f"including its features, benefits, and capabilities. Ignore comparisons and unrelated content.\n\n"
        f"{chunk_text}"
    )

def get_final_summary_prompt(product_name: str, summarized_chunks: str) -> str:
    return (
        f"Here are summarized chunks about '{product_name}'. Now, synthesize them into a single coherent summary "
        f"focusing only on the product’s features, use-cases, and benefits:\n\n"
        f"{summarized_chunks}"
    )

def get_cleaning_prompt(raw_text: str) -> str:
    return (
        "This transcript contains repeated lines or partially overlapping phrases. "
        "Please clean it by removing all repetition, filler lines, and irrelevant fragments. "
        "Preserve only meaningful and unique content:\n\n"
        f"{raw_text}"
    )
