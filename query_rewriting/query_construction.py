def query_construction(raw_utterance, conv_id, turn_id, query_config, history, title):
    """
    Parameters
    ----------
    raw_utterance - str original conversational query
    conv_id - conversation id
    turn_id - int, current turn
    query_config - QueryConfig, object with configurations to use in query rewriting
    history - str, what to prfix the qury with
    title - str, title of the conversation
    Returns
    -------
    str, utterance rewritten
    """
    if query_config.coreference_json:
        utterance = query_config.coreference_dic[str(conv_id)][str(turn_id)]
    else:
        utterance = raw_utterance
    if query_config.use_history and turn_id > 1:
        utterance = history + " " + utterance
    if query_config.use_title:
        utterance = title + " " + utterance
    return utterance
