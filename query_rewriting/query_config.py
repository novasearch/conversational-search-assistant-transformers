import json


class QueryConfig:
    """
    Class used to create a query configuration to use with run_test
    """

    def __init__(self, use_history, coreference_json, use_title, use_union, full_union):
        self.use_history = use_history
        self.coreference_json = coreference_json
        
        if coreference_json:
            with open(coreference_json, 'r') as json_file:
                self.coreference_dic = json.load(json_file)
        else:
            self.coreference_dic = None

        self.use_title = use_title
        self.use_union = use_union
        self.full_union = full_union

        if not use_union and full_union:
            raise ValueError("If using full_union=True must also use use_union=True")


# only the ones in the paper are left uncommented
query_configurations_2019 = {
    "ORIG": QueryConfig(use_history=False, coreference_json=False, use_title=False, use_union=False, full_union=False),
    "MANUAL": QueryConfig(use_history=False, coreference_json="./coreference_resolved_files/original_resolved.json",
                          use_title=False, use_union=False, full_union=False),

    # "H": QueryConfig(use_history=True, coreference_json=False, use_title=False, use_union=False, full_union=False),
    # "T": QueryConfig(use_history=False, coreference_json=False, use_title=True, use_union=False, full_union=False),
    "T5": QueryConfig(use_history=False,
                      coreference_json="./coreference_resolved_files/trec_cast_complete_t5_real_time_v2.json",
                      use_title=False, use_union=False, full_union=False),
    # "H_T5": QueryConfig(use_history=True,
    #                    coreference_json="./coreference_resolved_files/trec_cast_complete_t5_real_time_v2.json",
    #                    use_title=False, use_union=False, full_union=False),
    # "T5_UNION": QueryConfig(use_history=False,
    #                        coreference_json="./coreference_resolved_files/trec_cast_complete_t5_real_time_v2.json",
    #                        use_title=False, use_union=True, full_union=False),
}
