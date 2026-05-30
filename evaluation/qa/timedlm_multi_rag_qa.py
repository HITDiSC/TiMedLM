# This file evaluates TiMedLM on open-ended QA with multi-round RAG.
# Author: TiMedLM contributors
# Date: 2026-05-30
# Copyright (c) 2026 TiMedLM contributors. All rights reserved.
# See LICENSE file in the project root for license information.
# -*- coding: utf-8 -*-
"""
褰撳墠妯″瀷闂瓟棰樻祴璇曡剼鏈細Qwen3-8B + LoRA + 澶氳疆RAG + 鑷姩鎸囨爣 + 鏂偣淇濆瓨

闂瓟浼樺寲鐗?v3.5 绋冲畾鐗堬細
1. gold_evidence_ids 浼樺厛浠?reference 涓娊鍙?card_id
2. 鑷姩鎸囨爣璁＄畻鍓嶆竻娲?card_id
3. Evidence Support Rate 鏀逛负 Citation Validity锛屼笉澶稿ぇ涓哄够瑙夌巼
4. 闂瓟棰?top-k 浠?6 鎻愰珮鍒?10锛屾彁楂樺璇佹嵁瑕嗙洊
5. query prompt 寮哄寲锛氭寜瀛愰棶棰樻媶鍒嗘绱?
6. 妯℃澘 query 浣跨敤鈥滈骞叉牳蹇冨疄浣?+ 瀛愰棶棰樻ā鏉库€?
7. 淇鍧忓疄浣撴娊鍙栵細
   - 杩囨护鈥滅粨鍚堣棌鍖荤悊璁哄垎鏋愬叾鑽€т笌鐥呪€?
   - 杩囨护鈥滃姞閲嶇殑鐥団€?
   - 杩囨护鈥滃湪钘忓尰瀛︿腑鈥?
   - 杩囨护鈥滆缁撳悎璧ゅ反鐥呪€?
   - 杩囨护鈥滈ギ椋熲€濃€滃娌烩€濃€滄斁琛€鈥濃€滅伀鐏糕€濃€滆壘鐏糕€濈瓑娌荤枟鏂瑰紡浼疄浣?
8. 濡傛灉棰樺共宸叉湁鐤剧梾鍚?鑽墿鍚?鏂瑰墏鍚嶏紝鍒欎紭鍏堝彧鐢ㄨ繖浜涘疄浣擄紝涓嶅啀鐢ㄧ棁鐘舵垨娌荤枟鏂瑰紡鎵╁睍 query
9. 鍘绘帀 v4 涓繃寮虹殑鍙嶈瘉 query 鍜屽弽璇?rerank bonus锛岄伩鍏嶆硾鍖栧櫔澹?
10. final prompt 寮哄埗鍙兘寮曠敤妫€绱㈢粨鏋滀腑鐪熷疄鍑虹幇杩囩殑 card_id
11. 鏂板 Hit@10 / Recall@10
12. 鏂板 Hit@All / Recall@All锛岀敤浜庡尯鍒嗏€滃畬鍏ㄦ病妫€鍒扳€濆拰鈥滄鍒颁簡浣嗘病杩?top10鈥?
13. 鏂板 Gold Citation Recall
14. 鏀寔鏂偣淇濆瓨涓庢仮澶?

鏈増鏂板淇锛?
15. BERTScore 浣跨敤鏈湴 TIMEDLM_BERTSCORE_MODEL_TYPE
16. BERTScore 鏄惧紡鎸囧畾 num_layers=12锛岄伩鍏?KeyError
17. BERTScore 鍓嶅 prediction/reference 鎸?tokenizer 鎴柇鍒?510 tokens锛岄伩鍏嶈秴杩?512 鎶ラ敊
18. summary 鍜?config 涓褰?BERTScore 妯″瀷銆佸眰鏁般€佹埅鏂暱搴﹀拰鎴柇鏁伴噺
"""

import os
import re
import sys
import json
import math
import string
import torch
from datetime import datetime
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set

from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 璺緞閰嶇疆鍖猴細鍙渶瑕佹敼杩欓噷
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
MODEL_PATH = os.environ.get("TIMEDLM_MODEL_PATH", "models/timedlm-sft-v5")
LORA_PATH = os.environ.get("TIMEDLM_LORA_PATH", "models/timedlm-lora")
# 闂瓟娴嬭瘯闆嗚矾寰?
QA_TEST_PATH = os.environ.get("TIMEDLM_QA_TEST_PATH", "data/samples/oqa_eval_sample.json")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

RESULT_DIR = os.environ.get("TIMEDLM_QA_RESULT_DIR", "results/qa/timedlm_multi_rag")
RESULT_PATH = f"{RESULT_DIR}/qa_eval__qaandmcq_grpo_targeted290_rag_v5_{timestamp}.json"
CKPT_PATH = f"{RESULT_DIR}/qa_eval__qaandmcq_grpo_targeted280_rag_v5_ckpt.json"

# 鍙€夛細瀹屾暣鐭ヨ瘑搴?card 鏂囦欢锛岀敤浜庢鏌?citation existence
# 濡傛灉娌℃湁锛屼繚鎸?None锛岃剼鏈細鐢ㄦ湰娆℃绱㈠埌鐨?card_id 浣滀负 fallback
KNOWLEDGE_CARDS_PATH = None
# KNOWLEDGE_CARDS_PATH = "data/knowledge_cards.jsonl"
# KNOWLEDGE_CARDS_PATH = "data/knowledge_cards.json"


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 杩愯鍙傛暟
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

MAX_ROUNDS = 3

# 闂瓟棰橀渶瑕佸璇佹嵁锛宼op-k 寤鸿姣旈€夋嫨棰樺ぇ
RETRIEVE_TOP_K_PER_QUERY = 5
MAX_CARDS_PER_ROUND = 10

MAX_NEW_TOKENS_PLAN = 512
MAX_NEW_TOKENS_QUERY = 768
MAX_NEW_TOKENS_JUDGE = 512
MAX_NEW_TOKENS_ANSWER = 1400

SAVE_EVERY = 10
DEBUG_SAMPLES = 3

# 闂瓟 judge 闃堝€?
QA_HIGH_SCORE = 0.88
QA_VERY_HIGH_SCORE = 0.94

# 鏄惁璁＄畻 BERTScore
USE_BERTSCORE = True

# BERTScore 鏈湴妯″瀷閰嶇疆
# 娉ㄦ剰锛氭湰鍦版ā鍨嬭矾寰勫繀椤婚厤鍚?num_layers 浣跨敤锛屽惁鍒?bert_score 浼?KeyError
BERTSCORE_MODEL_TYPE = os.environ.get("TIMEDLM_BERTSCORE_MODEL_TYPE", "hfl/chinese-roberta-wwm-ext")

# chinese-roberta-wwm-ext 鏄?base 缁撴瀯锛屼竴鑸负 12 灞?
BERTSCORE_NUM_LAYERS = 12

# BERT/RoBERTa 鏈€澶?512 token锛岄鐣?special tokens
MAX_BERT_TOKENS = 510

os.makedirs(RESULT_DIR, exist_ok=True)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 妫€绱㈡ā鍧?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

RETRIEVAL_ROOT = os.environ.get(
    "RETRIEVAL_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "timedlm", "retrieval")),
)
sys.path.append(RETRIEVAL_ROOT)
from retrieval import retrieve_with_scores


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# card_id 姝ｅ垯
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

CARD_ID_RE = re.compile(
    r"\b(?:fact_[a-zA-Z0-9]+_\d{3}_\d{3}|case_[a-zA-Z0-9]+_\d{3}_\d{3}|diag_manual_\d{3})\b"
)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# System Prompt
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

SYSTEM_PROMPT = """\
浣犳槸涓€涓潰鍚戣棌鍖荤煡璇嗛棶绛旂殑AI鍔╂墜锛岃兘澶熷洖绛斿叧浜庤棌鍖荤悊璁恒€佽瘖鏂€佽嵂鐗┿€佹不鐤楃瓑鏂归潰鐨勯棶棰樸€備綘蹇呴』涓ユ牸閬靛惊浠ヤ笅瑙勫垯銆?

銆愭绱㈣鍒欍€?
鍥炵瓟鍓嶅繀椤诲厛妫€绱㈣棌鍖荤煡璇嗗簱銆?
涓嶈鍦ㄦ病鏈夋绱㈣瘉鎹殑鎯呭喌涓嬬洿鎺ョ粰鍑烘渶缁堝洖绛斻€?

銆愯瘖鏂帹鐞嗚鍒欍€?
褰撻棶棰樻秹鍙婄柧鐥呰瘖鏂椂锛屽繀椤讳弗鏍兼寜浠ヤ笅椤哄簭鎺ㄧ悊锛?
绗竴姝ワ細涓夎瘖鍒濇鍒ゆ柇锛堟湜璇?鑴夎瘖/闂瘖锛夛紝鑳界‘璇婂垯杩涘叆缁撹锛?
绗簩姝ワ細鑻ヤ笁璇婁笉鑳界‘璇婏紝鍚敤鍗佹柟闈㈢簿鍑嗚瘖鏂紱
绗笁姝ワ細鑻ュ崄鏂归潰浠嶄笉鑳界‘璇婏紝鍚敤浜旀閴村埆娉曪紱
绗洓姝ワ細缁撳悎鍥涚鐤剧梾鎬ц川杈撳嚭鏈€缁堢粨璁恒€?

銆愭绱㈡祦绋嬨€?
绗竴姝ワ細杈撳嚭 <plan>...</plan>锛岃鏄庨渶瑕佹绱粈涔堢煡璇嗐€?
绗簩姝ワ細杈撳嚭 <query>...</query>锛屽涓绱㈣瘝鐢ㄥ垎鍙峰垎闅斻€?
绗笁姝ワ細鏀跺埌妫€绱㈢粨鏋滃悗锛岃緭鍑?<judge>...</judge> 鍒ゆ柇璇佹嵁鏄惁鍏呭垎銆?
绗洓姝ワ細璇佹嵁鍏呭垎鍚庯紝鍩轰簬妫€绱㈢粨鏋滃洖绛旈棶棰樸€?

銆恞uery鐢熸垚瑕佹眰銆?
濡傛灉闂鍖呭惈澶氫釜鏂归潰锛屽繀椤讳负姣忎釜鏂归潰鍒嗗埆鐢熸垚妫€绱㈣瘝锛屼笉瑕佸彧鐢熸垚涓€涓缁焣uery銆?
query 搴斾紭鍏堝寘鍚骞蹭腑鐨勬牳蹇冪柧鐥呭悕銆佽嵂鐗╁悕銆佹柟鍓傚悕鎴栫梾鐥囧悕銆?
涓嶈鐢熸垚鈥滆棌鍖绘不鐤楁柟娉曗€濃€滆棌鍖昏瘖鏂柟娉曗€濊繖绫昏繃娉?query銆?
璇婃柇棰樿嚦灏戣鐩栵細鐥囩姸銆佽剦璞°€佸翱璞°€佺柧鐥呮€ц川銆侀壌鍒瘖鏂€?
鑽墿棰樿嚦灏戣鐩栵細鑽€с€佸姛鏁堛€侀€傚簲鐥囥€佺蹇屻€?
娌荤枟棰樿嚦灏戣鐩栵細楗鐤楁硶銆佽嵂鐗╃枟娉曘€佸娌绘硶銆佹敞鎰忎簨椤广€?
澶氫釜query蹇呴』鐢ㄥ垎鍙峰垎闅斻€?

銆愬洖绛旇姹傘€?
鏈€缁堝洖绛斿繀椤诲熀浜庢绱㈠埌鐨勮棌鍖诲吀绫嶅唴瀹广€?
濡傛灉妫€绱㈣瘉鎹腑鍖呭惈 card_id锛岃鍦ㄥ洖绛旀湯灏剧敤鈥滃紩鐢ㄦ潵婧愶細...鈥濆垪鍑轰娇鐢ㄥ埌鐨?card_id銆?
鍙兘寮曠敤涓婃柟妫€绱㈢粨鏋滀腑鐪熷疄鍑虹幇杩囩殑 card_id銆?
涓嶅緱鏀瑰啓銆佺寽娴嬨€佽ˉ鍏ㄦ垨缂栭€?card_id銆?
涓嶈缂栭€犱笉瀛樺湪鐨勫吀绫嶆潵婧愭垨 card_id銆?
"""


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鍔犺浇妯″瀷
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

print("鍔犺浇 tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("鍔犺浇鍩哄骇妯″瀷...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("鍔犺浇 LoRA 鏉冮噸...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

print("妯″瀷鍔犺浇瀹屾垚\n")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鐢熸垚鍑芥暟
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def generate(
    messages: List[Dict],
    stop_at: Optional[List[str]] = None,
    max_new_tokens: int = 1024,
    enable_thinking: bool = True,
    repetition_penalty: float = 1.05,
) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = outputs[0][inputs["input_ids"].shape[1]:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    response = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True
    ).strip()

    if stop_at:
        for tag in stop_at:
            if tag in response:
                response = response[:response.index(tag) + len(tag)]
                break

    return response


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鍩虹宸ュ叿鍑芥暟
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

CONTROL_TAGS = ["<plan>", "</plan>", "<query>", "</query>", "<judge>", "</judge>"]


def extract_tag(text: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_card_ids(text: str) -> List[str]:
    if not text:
        return []
    return list(dict.fromkeys(CARD_ID_RE.findall(text)))


def remove_card_ids_for_metric(text: str) -> str:
    if not text:
        return ""

    text = CARD_ID_RE.sub("[CARD_ID]", text)
    text = re.sub(r"寮曠敤鏉ユ簮[:锛歖\s*\[?.*?\]?\s*$", "", text, flags=re.S)
    text = re.sub(r"citations?[:锛歖\s*\[?.*?\]?\s*$", "", text, flags=re.I | re.S)

    return text.strip()


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\s+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[锛屻€傦紒锛燂紱锛氣€溾€濃€樷€欍€侊紙锛夈€娿€嬨€愩€慭[\]]", "", text)
    return text


def tokenize_zh(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    try:
        import jieba
        return [w for w in jieba.lcut(text) if w.strip()]
    except Exception:
        return list(normalize_text(text))


def get_card_content(card: Dict) -> str:
    evidence = card.get("evidence", {}) or {}
    return (
        card.get("content")
        or card.get("refined_result")
        or evidence.get("citation_text")
        or ""
    )


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# gold evidence 鎶藉彇
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def get_gold_evidence_ids(item: Dict) -> Set[str]:
    ids = set()

    ref = item.get("reference", "") or item.get("answer", "")
    ids.update(extract_card_ids(ref))

    for k in ["citations", "gold_card_ids"]:
        vals = item.get(k, [])
        if isinstance(vals, list):
            ids.update(str(x) for x in vals if x)
        elif isinstance(vals, str) and vals:
            ids.update(extract_card_ids(vals))
            if CARD_ID_RE.fullmatch(vals):
                ids.add(vals)

    if not ids:
        vals = item.get("seed_card_ids", [])
        if isinstance(vals, list):
            ids.update(str(x) for x in vals if x)
        elif isinstance(vals, str) and vals:
            ids.add(vals)

    return ids


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鍏抽敭璇?/ 瀹炰綋鎶藉彇
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

STOP_WORDS = {
    "?", "??", "??", "??", "??", "??", "??", "??",
    "??", "??", "??", "??", "??", "??", "??", "??",
    "??", "??", "??", "??", "??", "??", "??", "??",
    "??", "??", "??", "??", "??", "????", "??",
    "???", "??", "??", "??", "??", "??",
    "??", "???", "?????", "??????",
}

PSEUDO_ENTITY_WORDS = {
    "??", "??", "??", "??", "??", "??", "??",
    "????", "????", "???", "????",
    "????", "????", "????", "??", "??", "??",
}

ENTITY_STOP_WORDS = STOP_WORDS | PSEUDO_ENTITY_WORDS | {
    "??", "??", "???", "??",
    "????", "????", "????",
    "??", "??", "??", "??", "??", "??", "????",
    "????", "?????", "?????",
}

BAD_ENTITY_PATTERNS = [
    "??", "??", "??", "??", "??", "??", "??",
    "?????", "??????", "???", "????",
    "????", "????", "????", "??",
    "??", "??", "??", "??", "????",
    "????", "???", "????", "?????",
]

def is_bad_entity(entity: str) -> bool:
    e = (entity or "").strip()

    if len(e) < 2:
        return True

    # 澶暱鐨勪竴鑸槸鍗婃埅棰樺共锛屼笉鏄疄浣?
    if len(e) > 10:
        return True

    if e in ENTITY_STOP_WORDS:
        return True

    if any(p in e for p in BAD_ENTITY_PATTERNS):
        return True

    if e in {"藏医", "理论", "患者", "疾病", "症状", "方法"}:
        return True

    return False


def extract_question_terms(question: str) -> List[str]:
    text = question or ""
    text = re.sub(r"[锛屻€傦紱銆侊細:锛燂紒\?\s]+", "|", text)
    terms = []

    for t in text.split("|"):
        t = t.strip()
        if len(t) < 2:
            continue
        if t in STOP_WORDS:
            continue
        terms.append(t)

    return list(dict.fromkeys(terms))


def extract_core_entities(question: str, max_entities: int = 4) -> List[str]:
    """Extract short Chinese terms for retrieval query construction."""
    q = question or ""
    candidates: List[str] = []

    known_terms = [
        "?????", "??", "??????", "???",
        "?????", "???", "??", "???", "???",
        "?", "??", "??", "??", "??", "??", "???",
    ]
    for term in known_terms:
        if term in q and not is_bad_entity(term):
            candidates.append(term)

    suffix_patterns = [
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}??",
        r"[\u4e00-\u9fff]{2,8}??",
        r"[\u4e00-\u9fff]{2,8}???",
        r"[\u4e00-\u9fff]{2,8}??",
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}?",
        r"[\u4e00-\u9fff]{2,8}?",
    ]
    for pattern in suffix_patterns:
        for match in re.findall(pattern, q):
            if not is_bad_entity(match):
                candidates.append(match)

    symptom_terms = [
        "??", "???", "????", "??", "??", "??",
        "??", "??", "??", "??", "????", "????",
        "????", "????", "???", "????",
    ]
    for term in symptom_terms:
        if term in q and not is_bad_entity(term):
            candidates.append(term)

    for term in extract_question_terms(q):
        term = term.strip()
        if 2 <= len(term) <= 8 and not is_bad_entity(term):
            candidates.append(term)

    cleaned: List[str] = []
    for candidate in candidates:
        candidate = candidate.strip("?????:?????()??[]??")
        if candidate and not is_bad_entity(candidate) and candidate not in cleaned:
            cleaned.append(candidate)

    final = []
    for candidate in cleaned:
        if any(candidate != other and candidate in other for other in cleaned):
            continue
        final.append(candidate)

    return final[:max_entities]


def keyword_hit_in_cards(question: str, cards: List[Dict]) -> bool:
    terms = extract_question_terms(question)
    if not terms:
        return False

    card_text = "\n".join(
        (c.get("title", "") + " " + get_card_content(c))
        for c in cards
    )

    hit_count = 0
    for t in terms:
        if t and t in card_text:
            hit_count += 1

    return hit_count >= 1


def fact_card_count(cards: List[Dict]) -> int:
    return sum(1 for c in cards if c.get("card_type") == "fact")


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# query 鎵╁睍锛歷3.5 绋冲畾鐗?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def build_extra_queries(question: str, question_type: str = "") -> List[str]:
    """Build a small set of targeted retrieval queries for QA."""
    q = question or ""
    qt = question_type or ""
    entities = extract_core_entities(q, max_entities=4)
    if not entities:
        entities = [fallback_query_from_question(q)]

    is_diag = qt == "diagnostic" or any(
        term in q for term in ["??", "??", "??", "??", "??", "??", "??", "??", "??"]
    )
    is_drug = any(
        term in q for term in ["?", "??", "??", "??", "???", "??", "??", "??", "??"]
    )
    is_treatment = any(
        term in q for term in ["??", "??", "??", "??", "??", "??", "??", "????"]
    )
    is_match_judgement = any(
        term in q for term in ["????", "????", "????", "??", "???", "??", "??", "????"]
    )

    extra: List[str] = []
    for entity in entities[:3]:
        if is_bad_entity(entity):
            continue
        if is_diag:
            extra.extend([
                f"{entity} ?? ???? ?? ??",
                f"{entity} ?? ???? ??",
            ])
        if is_drug:
            extra.extend([
                f"{entity} ?? ?? ???",
                f"{entity} ?? ?? ???",
            ])
        if is_treatment:
            extra.extend([
                f"{entity} ???? ????",
                f"{entity} ??? ?? ?? ??",
                f"{entity} ???? ????",
            ])
        if is_match_judgement:
            extra.append(f"{entity} ?? ?? ???")

    cleaned: List[str] = []
    for query in extra:
        query = query.strip()
        if query and query not in cleaned:
            cleaned.append(query)
    return cleaned[:5]


def merge_query_content(model_query: str, question: str, question_type: str = "") -> str:
    queries = [
        query.strip()
        for query in re.split(r"[?;?,]", model_query or "")
        if query.strip()
    ]

    filtered_queries: List[str] = []
    bad_fragments = ["????????", "????", "?????", "?????"]
    for query in queries:
        if len(query) > 90:
            continue
        if any(fragment in query for fragment in bad_fragments):
            continue
        if query not in filtered_queries:
            filtered_queries.append(query)

    if not filtered_queries:
        filtered_queries = [fallback_query_from_question(question)]

    all_queries = filtered_queries + build_extra_queries(question, question_type)
    all_queries = list(dict.fromkeys(all_queries))[:8]
    return "?".join(all_queries)


def format_cards(cards: List[Dict]) -> str:
    results = []

    for card in cards:
        evidence = card.get("evidence", {}) or {}
        content = get_card_content(card)

        results.append({
            "card_id": card.get("card_id", ""),
            "title": card.get("title", ""),
            "card_type": card.get("card_type", ""),
            "content": content,
            "citation_text": evidence.get("citation_text", ""),
        })

    return json.dumps(results, ensure_ascii=False)


def rerank_cards_for_qa(question: str, scored_cards: List[Tuple[Dict, float]]):
    terms = extract_question_terms(question)
    entities = extract_core_entities(question, max_entities=4)
    reranked = []

    for card, score in scored_cards:
        title = card.get("title", "") or ""
        content = get_card_content(card)
        text = f"{title} {content}"
        cid = card.get("card_id", "") or ""
        ctype = card.get("card_type", "") or ""

        bonus = 0.0
        for term in terms:
            if term in title:
                bonus += 0.03
            elif term in text:
                bonus += 0.01
        for entity in entities:
            if is_bad_entity(entity):
                continue
            if entity in title:
                bonus += 0.05
            elif entity in text:
                bonus += 0.02
        if ctype == "fact":
            bonus += 0.015
        if cid.startswith("diag_manual") and not any(
            term in question for term in ["??", "??", "??", "??", "??", "????"]
        ):
            bonus -= 0.03

        adjusted = max(0.0, min(1.0, float(score) + bonus))
        reranked.append((card, adjusted, float(score), float(bonus)))

    reranked.sort(key=lambda item: item[1], reverse=True)
    return reranked


def retrieve_cards_by_query(query_content: str, question: str = "", verbose: bool = False):
    queries = [
        q.strip()
        for q in re.split(r"[锛?]", query_content)
        if q.strip()
    ]

    all_cards_scored = []

    for q in queries:
        try:
            all_cards_scored.extend(
                retrieve_with_scores(q, top_k=RETRIEVE_TOP_K_PER_QUERY)
            )
        except Exception as e:
            if verbose:
                print(f"銆愭绱㈠け璐ャ€憅uery={q} error={e}")

    seen = set()
    uniq_scored = []

    for card, score in all_cards_scored:
        cid = card.get("card_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            uniq_scored.append((card, float(score)))

    reranked = rerank_cards_for_qa(question, uniq_scored)

    top_scored = reranked[:MAX_CARDS_PER_ROUND]
    cards_this_round = [c for c, _, _, _ in top_scored]
    best_score = max((s for _, s, _, _ in top_scored), default=0.0)

    return reranked, top_scored, cards_this_round, best_score


def fallback_query_from_question(question: str) -> str:
    text = (question or "").strip()
    text = text.replace("??????????", "")
    text = text.replace("????????", "")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return "??????"
    text = re.split(r"[?;?.??]", text)[0].strip()
    return text[:80]


def is_judge_sufficient(
    judge_content: str,
    best_score: float,
    round_num: int,
    question: str,
    cards: List[Dict],
) -> bool:
    judge_content = judge_content or ""
    negative_words = {"???", "??", "??", "???", "????", "??", "????", "??", "????", "??", "???"}
    positive_words = {"????", "????", "????", "????", "??"}

    has_negative = any(word in judge_content for word in negative_words)
    has_positive = any(word in judge_content for word in positive_words)
    if has_positive and not has_negative:
        return True

    hit = keyword_hit_in_cards(question, cards)
    n_fact = fact_card_count(cards)
    if round_num == 1:
        return best_score >= QA_VERY_HIGH_SCORE and hit and n_fact >= 3
    if best_score >= QA_HIGH_SCORE and hit and n_fact >= 2:
        return True
    if best_score >= QA_VERY_HIGH_SCORE and n_fact >= 2:
        return True
    return False


def get_allowed_citation_ids(rounds_log: List[Dict]) -> List[str]:
    ids = []
    for r in rounds_log:
        for c in r.get("retrieved_cards", []):
            cid = c.get("card_id")
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def filter_answer_citations(answer: str, allowed_ids: List[str]) -> str:
    """Remove hallucinated card IDs from the answer citation area."""
    if not answer:
        return answer

    allowed_set = set(allowed_ids)
    bad_ids = [card_id for card_id in extract_card_ids(answer) if card_id not in allowed_set]
    if not bad_ids:
        return answer

    fixed = answer
    for card_id in bad_ids:
        fixed = fixed.replace(card_id, "")
    fixed = re.sub(r"[?,]\s*[?,]+", "?", fixed)
    if allowed_ids and re.search(r"????[:?]\s*$", fixed):
        fixed = re.sub(r"????[:?]\s*$", "?????" + "?".join(allowed_ids[:4]), fixed)
    return fixed.strip()


def rag_answer_qa(question: str, question_type: str = "", verbose: bool = False):
    """Answer an open-ended QA item with iterative retrieval."""
    rounds_log: List[Dict] = []
    query_content = fallback_query_from_question(question)
    final_cards: List[Dict] = []

    for round_idx in range(1, MAX_ROUNDS + 1):
        reranked, top_scored, cards_this_round, best_score = retrieve_cards_by_query(
            query_content,
            question=question,
            verbose=verbose,
        )
        final_cards = cards_this_round or final_cards
        rounds_log.append({
            "round": round_idx,
            "query": query_content,
            "best_score": best_score,
            "retrieved_cards": [card for card, _, _, _ in top_scored],
        })

        evidence_text = format_cards(cards_this_round)
        if round_idx >= MAX_ROUNDS or is_judge_sufficient("", best_score, round_idx, question, cards_this_round):
            break

        query_prompt = (
            "Generate focused retrieval queries for the next round. Return only query terms separated by semicolons.\n"
            f"Question: {question}\n"
            f"Current query: {query_content}\n"
            f"Retrieved evidence: {evidence_text[:3000]}"
        )
        query_response = generate([
            {"role": "system", "content": "You generate concise retrieval queries."},
            {"role": "user", "content": query_prompt},
        ], max_new_tokens=MAX_NEW_TOKENS_QUERY)
        query_content = merge_query_content(query_response, question, question_type)

    allowed_ids = get_allowed_citation_ids(rounds_log)
    evidence_text = format_cards(final_cards)
    answer_prompt = (
        "Answer the question using the retrieved evidence. Cite card_id values when useful.\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Return the final answer directly."
    )
    answer = generate([
        {"role": "system", "content": "You are a Tibetan medicine QA assistant. Answer using retrieved evidence."},
        {"role": "user", "content": answer_prompt},
    ], max_new_tokens=MAX_NEW_TOKENS_ANSWER)
    answer = filter_answer_citations(answer, allowed_ids)
    return answer, rounds_log, "multi_rag"


def lcs_length(x: List[str], y: List[str]) -> int:
    if not x or not y:
        return 0

    m, n = len(x), len(y)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if x[i - 1] == y[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]


def rouge_l_score(pred: str, ref: str) -> float:
    pred = remove_card_ids_for_metric(pred)
    ref = remove_card_ids_for_metric(ref)

    pred_tokens = tokenize_zh(pred)
    ref_tokens = tokenize_zh(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = lcs_length(pred_tokens, ref_tokens)

    recall = lcs / len(ref_tokens)
    precision = lcs / len(pred_tokens)

    if recall + precision == 0:
        return 0.0

    beta = precision / (recall + 1e-12)
    score = ((1 + beta ** 2) * precision * recall) / (
        recall + beta ** 2 * precision + 1e-12
    )

    return float(score)


def ngram_counts(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def bleu4_score(pred: str, ref: str) -> float:
    pred = remove_card_ids_for_metric(pred)
    ref = remove_card_ids_for_metric(ref)

    pred_tokens = tokenize_zh(pred)
    ref_tokens = tokenize_zh(ref)

    if not pred_tokens or not ref_tokens:
        return 0.0

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []

    for n in range(1, 5):
        pred_ngrams = ngram_counts(pred_tokens, n)
        ref_ngrams = ngram_counts(ref_tokens, n)

        if not pred_ngrams:
            precisions.append(1e-9)
            continue

        overlap = 0
        total = sum(pred_ngrams.values())

        for ng, count in pred_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))

        precisions.append((overlap + 1) / (total + 1))

    log_precision = sum(w * math.log(p) for w, p in zip(weights, precisions))

    pred_len = len(pred_tokens)
    ref_len = len(ref_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = math.exp(1 - ref_len / max(pred_len, 1))

    return float(bp * math.exp(log_precision))


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# RAG 鎸囨爣锛欻it / Recall
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def get_retrieved_card_ids(rounds_log: List[Dict]) -> List[str]:
    ids = []
    for r in rounds_log:
        for c in r.get("retrieved_cards", []):
            cid = c.get("card_id")
            if cid and cid not in ids:
                ids.append(cid)
    return ids


def compute_hit_recall_at_k(
    retrieved_ids: List[str],
    gold_ids: Set[str],
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    if not gold_ids:
        return None, None

    top_ids = retrieved_ids[:k]
    retrieved_set = set(top_ids)

    hit = 1.0 if retrieved_set & gold_ids else 0.0
    recall = len(retrieved_set & gold_ids) / len(gold_ids)

    return hit, recall


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# Citation Validity
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def load_knowledge_cards(path: Optional[str]) -> Dict[str, str]:
    if not path or not os.path.exists(path):
        return {}

    card_map = {}

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            iterable = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                iterable.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            iterable = data.values()
        else:
            iterable = data

    for item in iterable:
        if not isinstance(item, dict):
            continue

        cid = item.get("card_id", "")
        if not cid:
            continue

        content = (
            item.get("content")
            or item.get("refined_result")
            or item.get("citation_text")
            or item.get("title")
            or ""
        )

        card_map[cid] = content

    print(f"Loaded knowledge cards: {len(card_map)}")
    return card_map


KNOWLEDGE_CARD_MAP = load_knowledge_cards(KNOWLEDGE_CARDS_PATH)


def build_retrieved_card_map(rounds_log: List[Dict]) -> Dict[str, str]:
    card_map = {}

    for r in rounds_log:
        for c in r.get("retrieved_cards", []):
            cid = c.get("card_id", "")
            content = c.get("content", "") or c.get("title", "")
            if cid:
                card_map[cid] = content

    return card_map


def compute_citation_validity(answer: str, rounds_log: List[Dict]) -> Dict:
    cited_ids = extract_card_ids(answer)
    retrieved_card_map = build_retrieved_card_map(rounds_log)
    retrieved_ids = set(retrieved_card_map.keys())

    if KNOWLEDGE_CARD_MAP:
        existence_map = KNOWLEDGE_CARD_MAP
    else:
        existence_map = retrieved_card_map

    if not cited_ids:
        return {
            "cited_ids": [],
            "citation_coverage": 0.0,
            "citation_existence_rate": None,
            "citation_from_retrieval_rate": None,
            "citation_validity": None,
        }

    existing_ids = [cid for cid in cited_ids if cid in existence_map]
    from_retrieval_ids = [cid for cid in cited_ids if cid in retrieved_ids]
    non_existing_ids = [cid for cid in cited_ids if cid not in existence_map]

    citation_existence_rate = len(existing_ids) / len(cited_ids)
    citation_from_retrieval_rate = len(from_retrieval_ids) / len(cited_ids)

    # 鏇翠弗鏍硷細鎵€鏈夊紩鐢ㄥ繀椤诲瓨鍦紝涓旀墍鏈夊紩鐢ㄩ兘蹇呴』鏉ヨ嚜鏈妫€绱?
    citation_validity = (
        len(non_existing_ids) == 0
        and len(from_retrieval_ids) == len(cited_ids)
    )

    return {
        "cited_ids": cited_ids,
        "citation_coverage": 1.0,
        "citation_existence_rate": citation_existence_rate,
        "citation_from_retrieval_rate": citation_from_retrieval_rate,
        "citation_validity": 1.0 if citation_validity else 0.0,
        "non_existing_cited_ids": non_existing_ids,
        "retrieved_cited_ids": from_retrieval_ids,
    }


def compute_gold_citation_recall(cited_ids: List[str], gold_ids: Set[str]) -> Optional[float]:
    if not gold_ids:
        return None

    if not cited_ids:
        return 0.0

    return len(set(cited_ids) & gold_ids) / len(gold_ids)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# BERTScore锛氬凡淇
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def token_truncate(text: str, bert_tokenizer, max_tokens: int = 510) -> str:
    """
    鎸?tokenizer 鎴柇锛岄伩鍏?BERTScore 杈撳叆瓒呰繃 512 token銆?
    """
    if not text:
        return ""

    ids = bert_tokenizer.encode(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )

    return bert_tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def compute_bertscore_batch(preds: List[str], refs: List[str]) -> Tuple[List[Optional[float]], int, int]:
    """
    杩斿洖锛?
    1. 姣忔潯鏍锋湰鐨?BERTScore-F1
    2. prediction 琚埅鏂殑鏁伴噺
    3. reference 琚埅鏂殑鏁伴噺
    """
    if not USE_BERTSCORE:
        return [None] * len(preds), 0, 0

    try:
        from bert_score import score
        from transformers import AutoTokenizer as BertAutoTokenizer
    except Exception as e:
        print(f"鏈畨瑁?bert_score 鎴?transformers锛岃烦杩?BERTScore銆俥rror={e}")
        return [None] * len(preds), 0, 0

    try:
        print("鍔犺浇 BERTScore tokenizer...")
        bert_tokenizer = BertAutoTokenizer.from_pretrained(
            BERTSCORE_MODEL_TYPE,
            use_fast=True,
        )

        clean_preds = []
        clean_refs = []

        pred_trunc_count = 0
        ref_trunc_count = 0

        for p, r in zip(preds, refs):
            p = remove_card_ids_for_metric(p)
            r = remove_card_ids_for_metric(r)

            p_ids = bert_tokenizer.encode(p, add_special_tokens=False)
            r_ids = bert_tokenizer.encode(r, add_special_tokens=False)

            if len(p_ids) > MAX_BERT_TOKENS:
                pred_trunc_count += 1
            if len(r_ids) > MAX_BERT_TOKENS:
                ref_trunc_count += 1

            clean_preds.append(
                token_truncate(p, bert_tokenizer, MAX_BERT_TOKENS)
            )
            clean_refs.append(
                token_truncate(r, bert_tokenizer, MAX_BERT_TOKENS)
            )

        print(f"BERTScore prediction 鎴柇鏁伴噺: {pred_trunc_count}")
        print(f"BERTScore reference 鎴柇鏁伴噺: {ref_trunc_count}")

        _, _, f1 = score(
            clean_preds,
            clean_refs,
            model_type=BERTSCORE_MODEL_TYPE,
            num_layers=BERTSCORE_NUM_LAYERS,
            verbose=True,
            rescale_with_baseline=False,
            batch_size=8,
        )

        return [float(x) for x in f1.cpu().tolist()], pred_trunc_count, ref_trunc_count

    except Exception as e:
        print(f"BERTScore 璁＄畻澶辫触锛岃烦杩囥€俥rror={e}")
        return [None] * len(preds), 0, 0


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 鏂偣淇濆瓨 / 鍔犺浇
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def load_checkpoint():
    if not os.path.exists(CKPT_PATH):
        return [], set()

    try:
        with open(CKPT_PATH, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        results = ckpt.get("results", [])
        done_ids = {str(x["sample_id"]) for x in results}

        print(f"[鏂偣鎭㈠] 宸插畬鎴?{len(results)} 鏉★紝缁х画璇勪及...\n")
        return results, done_ids

    except Exception as e:
        print(f"[鏂偣鎭㈠澶辫触] {e}锛屼粠澶村紑濮媆n")
        return [], set()


def save_checkpoint(results: List[Dict], total: int):
    ckpt = {
        "timestamp": timestamp,
        "model_path": MODEL_PATH,
        "lora_path": LORA_PATH,
        "qa_test_path": QA_TEST_PATH,
        "progress": f"{len(results)}/{total}",
        "results": results,
    }

    with open(CKPT_PATH, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False, indent=2)


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 姹囨€荤粺璁?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def safe_mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def summarize_results(results: List[Dict]) -> Dict:
    rouge_l = safe_mean([r.get("rouge_l") for r in results])
    bleu4 = safe_mean([r.get("bleu4") for r in results])
    bertscore_f1 = safe_mean([r.get("bertscore_f1") for r in results])

    hit6 = safe_mean([r.get("hit6") for r in results])
    recall6 = safe_mean([r.get("recall6") for r in results])
    hit10 = safe_mean([r.get("hit10") for r in results])
    recall10 = safe_mean([r.get("recall10") for r in results])
    hit_all = safe_mean([r.get("hit_all") for r in results])
    recall_all = safe_mean([r.get("recall_all") for r in results])

    citation_coverage = safe_mean([r.get("citation_coverage") for r in results])
    citation_existence_rate = safe_mean([r.get("citation_existence_rate") for r in results])
    citation_from_retrieval_rate = safe_mean([r.get("citation_from_retrieval_rate") for r in results])
    citation_validity = safe_mean([r.get("citation_validity") for r in results])

    gold_citation_recall = safe_mean([r.get("gold_citation_recall") for r in results])

    avg_rounds = safe_mean([r.get("retrieval_rounds") for r in results])
    source_dist = Counter(r.get("answer_source", "unknown") for r in results)

    return {
        "total": len(results),
        "rouge_l": round(rouge_l, 4) if rouge_l is not None else None,
        "bleu4": round(bleu4, 4) if bleu4 is not None else None,
        "bertscore_f1": round(bertscore_f1, 4) if bertscore_f1 is not None else None,

        "hit6": round(hit6, 4) if hit6 is not None else None,
        "recall6": round(recall6, 4) if recall6 is not None else None,
        "hit10": round(hit10, 4) if hit10 is not None else None,
        "recall10": round(recall10, 4) if recall10 is not None else None,
        "hit_all": round(hit_all, 4) if hit_all is not None else None,
        "recall_all": round(recall_all, 4) if recall_all is not None else None,

        "citation_coverage": round(citation_coverage, 4) if citation_coverage is not None else None,
        "citation_existence_rate": round(citation_existence_rate, 4) if citation_existence_rate is not None else None,
        "citation_from_retrieval_rate": round(citation_from_retrieval_rate, 4) if citation_from_retrieval_rate is not None else None,
        "citation_validity": round(citation_validity, 4) if citation_validity is not None else None,
        "gold_citation_recall": round(gold_citation_recall, 4) if gold_citation_recall is not None else None,

        "avg_rounds": round(avg_rounds, 2) if avg_rounds is not None else None,
        "answer_source_dist": dict(source_dist),
    }


# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
# 涓绘祦绋?
# 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    print(f"Loading QA test set: {QA_TEST_PATH}")
    with open(QA_TEST_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"QA test samples: {len(dataset)}")

    results, done_ids = load_checkpoint()
    for idx, item in enumerate(tqdm(dataset, desc="QA evaluation")):
        sample_id = str(item.get("id", item.get("question_id", idx + 1)))
        if sample_id in done_ids:
            continue

        question = item.get("input") or item.get("question") or item.get("query") or ""
        reference = item.get("reference") or item.get("answer") or ""
        question_type = item.get("question_type", "")
        if not question:
            print(f"Sample {sample_id} has no question; skipped.")
            continue

        verbose = len(results) < DEBUG_SAMPLES
        pred, rounds_log, answer_source = rag_answer_qa(question, question_type=question_type, verbose=verbose)
        retrieved_ids = get_retrieved_card_ids(rounds_log)
        gold_ids = get_gold_evidence_ids(item)
        hit6, recall6 = compute_hit_recall_at_k(retrieved_ids, gold_ids, k=6)
        hit10, recall10 = compute_hit_recall_at_k(retrieved_ids, gold_ids, k=10)
        hit_all, recall_all = compute_hit_recall_at_k(retrieved_ids, gold_ids, k=len(retrieved_ids))
        rouge_l = rouge_l_score(pred, reference) if reference else None
        bleu4 = bleu4_score(pred, reference) if reference else None
        citation_info = compute_citation_validity(pred, rounds_log)
        gold_citation_recall = compute_gold_citation_recall(citation_info.get("cited_ids", []), gold_ids)

        result = {
            "sample_id": sample_id,
            "question": question,
            "reference": reference,
            "prediction": pred,
            "question_type": question_type,
            "seed_card_ids": item.get("seed_card_ids", []),
            "citations": item.get("citations", []),
            "rouge_l": rouge_l,
            "bleu4": bleu4,
            "bertscore_f1": None,
            "hit6": hit6,
            "recall6": recall6,
            "hit10": hit10,
            "recall10": recall10,
            "hit_all": hit_all,
            "recall_all": recall_all,
            "gold_evidence_ids": list(gold_ids),
            "retrieved_card_ids": retrieved_ids,
            "citation_coverage": citation_info.get("citation_coverage"),
            "citation_existence_rate": citation_info.get("citation_existence_rate"),
            "citation_from_retrieval_rate": citation_info.get("citation_from_retrieval_rate"),
            "citation_validity": citation_info.get("citation_validity"),
            "gold_citation_recall": gold_citation_recall,
            "cited_ids": citation_info.get("cited_ids", []),
            "non_existing_cited_ids": citation_info.get("non_existing_cited_ids", []),
            "retrieved_cited_ids": citation_info.get("retrieved_cited_ids", []),
            "retrieval_rounds": len(rounds_log),
            "answer_source": answer_source,
            "rounds_log": rounds_log,
        }
        results.append(result)

        if verbose:
            print(f"Sample {sample_id}")
            print(pred[:1200])
            print(f"ROUGE-L={rouge_l}, BLEU-4={bleu4}, Hit@10={hit10}, Recall@10={recall10}")

        if len(results) % SAVE_EVERY == 0:
            save_checkpoint(results, len(dataset))
            tqdm.write(json.dumps(summarize_results(results), ensure_ascii=False, indent=2))

    if USE_BERTSCORE:
        preds = [r.get("prediction", "") for r in results]
        refs = [r.get("reference", "") for r in results]
        bert_scores, skipped, truncated = compute_bertscore_batch(preds, refs)
        for result, score in zip(results, bert_scores):
            result["bertscore_f1"] = score
    else:
        skipped = 0
        truncated = 0

    summary = summarize_results(results)
    summary["bertscore_skipped"] = skipped
    summary["bertscore_truncated"] = truncated
    output = {
        "config": {
            "model_path": MODEL_PATH,
            "lora_path": LORA_PATH,
            "qa_test_path": QA_TEST_PATH,
            "result_path": RESULT_PATH,
            "use_bertscore": USE_BERTSCORE,
            "bertscore_model_type": BERTSCORE_MODEL_TYPE if USE_BERTSCORE else None,
            "bertscore_num_layers": BERTSCORE_NUM_LAYERS if USE_BERTSCORE else None,
        },
        "summary": summary,
        "results": results,
    }
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)
    print(f"Saved results to: {RESULT_PATH}")

if __name__ == "__main__":
    main()
