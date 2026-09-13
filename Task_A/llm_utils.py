# Task_A/llm_utils.py


def extract_text_from_response(response) -> str:
    """
    Find the text block in a Claude response, rather than assuming content[0]
    is always the answer -- responses can include a ThinkingBlock before the
    TextBlock, which has no .text attribute and breaks a positional assumption.
    """
    for block in response.content:
        if block.type == "text":
            return block.text
    return None
