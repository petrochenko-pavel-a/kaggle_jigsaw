from musket_text.bert.load import load_google_bert
from musket_core.context import get_current_project_path

BERT_DIR = get_current_project_path() + "/data/bert/uncased_L-12_H-768_A-12"
BERT_MAX_SEQ_LENGTH = 256

def bert(input):
    g_bert, cfg = load_google_bert(base_location=BERT_DIR + '/', max_len=BERT_MAX_SEQ_LENGTH, use_attn_mask=False,customInputs=input)
    outputs = g_bert.outputs
    return outputs[0]