import re


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
