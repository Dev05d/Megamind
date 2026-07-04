import re

def get_sentence_overlap(text: str, num_sentences: int = 2) -> str:
    """Extracts the last N sentences for semantic overlapping."""
    # This regex looks for a period, question mark, or exclamation point, 
    # followed by whitespace, and then a capital letter, number, or LaTeX '$'.
    # It avoids splitting on decimals like '3.14' or acronyms like 'U.S.A.'
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9$#*-])', text.strip())
    
    if len(sentences) <= num_sentences:
        return text.strip()
        
    return " ".join(sentences[-num_sentences:])

def chunk_text(text: str, target: int = 1200, ceiling: int = 1600, overlap_sentences: int = 2) -> list[str]:
    """
    Elastic Semantic Chunker.
    Groups by paragraphs. Stretches to a ceiling to avoid mid-paragraph cuts. 
    Overlaps adjacent chunks using complete, logical sentences.
    """
    if not text or not text.strip():
        return []

    # 1. Break the document down into logical structural blocks
    paragraphs = re.split(r'\n\n+', text)
    
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        predicted_size = len(current_chunk) + len(para) + 2
        
        # 2. Comfortable Fit: Add it to the buffer and move on.
        if predicted_size <= target:
            current_chunk += ("\n\n" + para) if current_chunk else para
            continue
            
        # 3. Elastic Stretch: We crossed the target, but haven't hit the ceiling.
        if predicted_size <= ceiling:
            current_chunk += "\n\n" + para
            chunks.append(current_chunk.strip())
            
            # Seed the start of the next chunk with our complete sentences
            overlap = get_sentence_overlap(current_chunk, overlap_sentences)
            current_chunk = f"...{overlap}"
            continue
            
        # 4. Ceiling Breached: We must close the current chunk NOW.
        if current_chunk:
            chunks.append(current_chunk.strip())
            overlap = get_sentence_overlap(current_chunk, overlap_sentences)
            current_chunk = f"...{overlap}\n\n{para}"
        else:
            # Edge case: The incoming paragraph itself is massive.
            current_chunk = para

        # 5. Handle massive, unbroken text walls (Drop down to sentence-level splitting)
        while len(current_chunk) > ceiling:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9$#*-])', current_chunk)
            
            temp_size = 0
            split_idx = 0
            for i, s in enumerate(sentences):
                temp_size += len(s) + 1
                if temp_size >= target:
                    split_idx = i + 1
                    break
            
            # Fallback: If no sentence breaks exist (e.g., a massive code block), hard slice.
            if split_idx == 0 or split_idx == len(sentences):
                hard_cut = current_chunk[:target]
                chunks.append(hard_cut.strip())
                current_chunk = f"...{current_chunk[target:].strip()}"
            else:
                # Clean slice at the semantic sentence boundary
                safe_block = " ".join(sentences[:split_idx])
                chunks.append(safe_block.strip())
                
                overlap = get_sentence_overlap(safe_block, overlap_sentences)
                remaining = " ".join(sentences[split_idx:])
                current_chunk = f"...{overlap} {remaining}"

    # 6. Flush whatever is left in the final buffer
    if current_chunk and current_chunk != "...":
        chunks.append(current_chunk.strip())

    return chunks