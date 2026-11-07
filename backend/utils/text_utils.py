import re


def _split_long(sentence: str, limit: int) -> list[str]:
    """Break a sentence longer than ``limit`` into pieces of at most ``limit`` characters,
    cutting at spaces where possible (a single word longer than ``limit`` is cut mid-word)"""
    pieces: list[str] = []
    current = ""
    for word in sentence.split(" "):
        while len(word) > limit:  # no space to cut at: hard split
            if current:
                pieces.append(current)
                current = ""
            pieces.append(word[:limit])
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into sentence-based chunks of at most ``chunk_size`` characters, with
    roughly ``chunk_overlap`` characters of trailing sentences repeated at the start of the
    next chunk"""

    # Clean up the text
    text = re.sub(r"\s+", " ", text.strip())  # Normalize whitespace

    # Better sentence splitting that handles abbreviations
    # This regex looks for periods followed by whitespace and capital letters
    # but ignores common abbreviations
    sentence_endings = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\!|\?)\s+(?=[A-Z])")
    sentences = sentence_endings.split(text)

    # Clean sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # A sentence longer than chunk_size would otherwise become an oversized chunk
    if chunk_size > 0:
        sentences = [
            piece
            for s in sentences
            for piece in (_split_long(s, chunk_size) if len(s) > chunk_size else [s])
        ]

    chunks = []
    i = 0

    while i < len(sentences):
        current_chunk: list[str] = []
        current_size = 0

        # Build chunk starting from sentence i
        for j in range(i, len(sentences)):
            sentence = sentences[j]

            # Calculate size with space
            space_size = 1 if current_chunk else 0
            total_addition = len(sentence) + space_size

            # Check if adding this sentence would exceed chunk size
            if current_size + total_addition > chunk_size and current_chunk:
                break

            current_chunk.append(sentence)
            current_size += total_addition

        # Add chunk if we have content
        if current_chunk:
            chunks.append(" ".join(current_chunk))

            # Calculate overlap for next chunk
            if chunk_overlap > 0:
                # Find how many sentences to overlap
                overlap_size = 0
                overlap_sentences = 0

                # Count backwards from end of current chunk
                for k in range(len(current_chunk) - 1, -1, -1):
                    sentence_len = len(current_chunk[k]) + (1 if k < len(current_chunk) - 1 else 0)
                    if overlap_size + sentence_len <= chunk_overlap:
                        overlap_size += sentence_len
                        overlap_sentences += 1
                    else:
                        break

                # Move start position considering overlap
                next_start = i + len(current_chunk) - overlap_sentences
                i = max(next_start, i + 1)  # Ensure we make progress
            else:
                # No overlap - move to next sentence after current chunk
                i += len(current_chunk)
        else:
            # No sentences fit, move to next
            i += 1

    return chunks
