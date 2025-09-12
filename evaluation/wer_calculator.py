import jiwer


def calc_wer(pred, gt):
    """
    Calculate the Word Error Rate (WER) between a predicted transcription and a ground truth transcription.

    This function normalizes both the prediction and ground truth by converting to lowercase,
    removing punctuation, stripping whitespace, and reducing multiple spaces before computing WER.

    Args:
        pred (str): The predicted transcription.
        gt (str): The ground truth transcription.

    Returns:
        float: The word error rate (WER) as a float between 0.0 and 1.0.
    """
    transformation = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.ReduceToListOfListOfWords()
    ])
    measures = jiwer.process_words(
        gt,
        pred,
        reference_transform=transformation,
        hypothesis_transform=transformation
    )

    return measures


def main():
    # Example usage
    # measures = calc_wer(transcription, gt_transcript)
    measures = calc_wer("This is, a test!", "THIS IS A TEST")
    print(f"\nWord Error Rate: {100 * measures.wer:.3f} %")


if __name__ == "__main__":
    main()