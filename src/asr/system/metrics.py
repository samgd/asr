import jiwer


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    """Compute WER components between ``ref`` and ``hyp`` (edit count, reference word count)."""
    wo = jiwer.process_words(ref, hyp)
    return wo.substitutions + wo.deletions + wo.insertions, len(ref.split())
