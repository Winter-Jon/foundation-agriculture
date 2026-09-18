"""Public error-specific repair instructions; never supply private labels."""

def repair_instruction(errors):
    details = '; '.join(map(str,errors))
    if any(key in details for key in ('malformed_final','placeholder_analysis','duplicate_or_nested_final_tags','nested_final_tags')):
        action = ('Keep your existing observations, candidate comparisons, evidence, exclusions and uncertainty. '
                  'Move the complete substantive analysis inside a single <think> opening tag and </think> closing tag. '
                  'Then put your existing answer inside a single <answer>...</answer> block. '
                  'Do not discard the analysis or output the literal placeholder word analysis. '
                  'Do not write Analysis: or any prose outside the tags. This is a formatting correction only; do not change your diagnosis.')
    elif 'invalid_rag_arguments' in details:
        action = ('Call agrinet_rag_search with retrieval_type visual/name/semantic, a nonempty query and rationale, and top_k=3. '
                  'Visual automatically uses the bound query image; image may be omitted or equal query_image. '
                  'Name and semantic must omit image. Do not add any other arguments.')
    else:
        action = 'Follow the public tool schema and required classifier-before-RAG order; complete required tools before a final answer.'
    return 'Protocol repair only: ' + details + '. ' + action + ' No answer correction is supplied.'
