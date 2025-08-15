from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql import Row
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_ADLS = "adl://m365-consumer-cdp-dev-c15.azuredatalakestore.net"
REL_INPUT = "/local/CGE/Analysis/v-koushikbo/DS/MEI/local/CGE/CTA_PreProcessed_20250807.csv"
SRC_PATH  = f"{BASE_ADLS}{REL_INPUT}"
OUT_PATH  = f"{BASE_ADLS}/local/CGE/Analysis/v-koushikbo/DS/MEI/CTA_PreProcessed_20250807_enriched"

MODEL_BASE_PATH = "/mnt/models/cta_intents"
STAGE1_DIR = f"{MODEL_BASE_PATH}/stage1_coarse"
STAGE1_LABELS = f"{STAGE1_DIR}/labels.json"
STAGE2_DIRS = {
    "M365":    f"{MODEL_BASE_PATH}/stage2/m365",
    "Xbox":    f"{MODEL_BASE_PATH}/stage2/xbox",
    "Azure":   f"{MODEL_BASE_PATH}/stage2/azure",
    "Windows": f"{MODEL_BASE_PATH}/stage2/windows",
    "Surface": f"{MODEL_BASE_PATH}/stage2/surface",
    "Bing":    f"{MODEL_BASE_PATH}/stage2/bing",
    "Copilot": f"{MODEL_BASE_PATH}/stage2/copilot",
    "General": f"{MODEL_BASE_PATH}/stage2/general",
}

df = (spark.read
      .option("header", "true")
      .option("multiLine", "true")
      .csv(SRC_PATH))

CTA_COL = "Pre-Processed_1" if "Pre-Processed_1" in df.columns else ("CTA_Translated" if "CTA_Translated" in df.columns else None)
LOB_COL = "LOBs" if "LOBs" in df.columns else ("LOB" if "LOB" in df.columns else None)
assert CTA_COL, "CTA column not found"
assert LOB_COL, "LOB column not found"

df = df.withColumn(CTA_COL, F.trim(F.col(CTA_COL))).withColumn(LOB_COL, F.trim(F.col(LOB_COL)))

def derive_ms_category(lob, action, text):
    tl = (text or "").lower()
    l = (lob or "").lower()
    a = (action or "").lower()
    if "copilot" in tl and any(x in a for x in ["enable", "turn on", "try"]):
        return "Copilot Feature Activation"
    if "microsoft 365" in tl or "office" in tl or "m365" in l:
        if "start trial" in a: return "M365 Acquisition Campaign"
        if any(x in a for x in ["buy", "upgrade", "subscribe"]): return "M365 Upsell"
        if "renew" in a: return "M365 Retention"
        if "sign in" in a: return "Account Engagement (M365)"
        return "M365 Engagement"
    if "windows" in l or "windows" in tl:
        if any(x in a for x in ["download", "install"]): return "Windows Adoption"
        if any(x in a for x in ["enable", "turn on"]): return "Windows Feature Activation"
        return "Windows Engagement"
    if "xbox" in l or "xbox" in tl or "game pass" in tl:
        if "renew" in a: return "Xbox Retention"
        if any(x in a for x in ["buy", "upgrade", "subscribe"]): return "Xbox Upsell"
        if any(x in a for x in ["play", "watch"]): return "Xbox Engagement"
        return "Xbox Awareness"
    if "azure" in l or "azure" in tl:
        if "contact sales" in a: return "Azure Sales Assisted"
        if any(x in a for x in ["start trial", "try"]): return "Azure Trial Conversion"
        if "learn more" in a: return "Azure Solution Education"
        return "Azure Engagement"
    if "bing" in l or "bing" in tl:
        if "sign in" in a: return "Bing Account Engagement"
        return "Bing Engagement Campaign"
    if "surface" in l or "surface" in tl:
        if any(x in a for x in ["buy", "upgrade"]): return "Surface Device Purchase"
        return "Surface Engagement"
    if "learn more" in a: return "Education/Awareness"
    return "General Engagement"

def canonicalize_cta(text, lob, action):
    t = (text or "").strip()
    if not t: return t
    repl = {
        "sign in": "Sign in",
        "sign-in": "Sign in",
        "turn on": "Turn on",
        "enable": "Turn on",
        "buy now": "Buy now",
        "learn more": "Learn more",
        "start free trial": "Start your free trial",
        "download": "Download",
        "install": "Install",
        "renew": "Renew",
        "subscribe": "Subscribe",
        "try now": "Try now",
    }
    low = t.lower()
    for k, v in repl.items():
        low = low.replace(k, v.lower())
    canon = low[:1].upper() + low[1:]
    canon = canon.rstrip(".!? ")
    if len(canon) <= 18:
        l = (lob or "").lower()
        if "azure" in l and "learn more" in canon.lower():
            canon = "Learn more about Azure"
        elif ("m365" in l or "office" in l) and "start your free trial" in canon.lower():
            canon = "Start your Microsoft 365 free trial"
        elif ("xbox" in l) and "renew" in canon.lower():
            canon = "Renew your Xbox Game Pass"
    return canon

def _load_labels(path):
    with open(path, "r") as f:
        return json.load(f)

def _softmax(logits):
    return torch.nn.functional.softmax(logits, dim=-1)

def _batch(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

_GLOBAL_STAGE1 = {"model": None, "tok": None, "labels": None}
_GLOBAL_STAGE2 = {}

def _device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _get_stage1():
    if _GLOBAL_STAGE1["model"] is None:
        tok = AutoTokenizer.from_pretrained(STAGE1_DIR)
        mdl = AutoModelForSequenceClassification.from_pretrained(STAGE1_DIR)
        lbl = _load_labels(STAGE1_LABELS)
        _GLOBAL_STAGE1.update({"model": mdl, "tok": tok, "labels": lbl})
    return _GLOBAL_STAGE1["tok"], _GLOBAL_STAGE1["model"], _GLOBAL_STAGE1["labels"]

def _get_stage2(family):
    fam = family if family in STAGE2_DIRS else "General"
    if fam not in _GLOBAL_STAGE2:
        base = STAGE2_DIRS[fam]
        tok = AutoTokenizer.from_pretrained(base)
        mdl = AutoModelForSequenceClassification.from_pretrained(base)
        lbl = _load_labels(os.path.join(base, "labels.json"))
        _GLOBAL_STAGE2[fam] = {"tok": tok, "model": mdl, "labels": lbl}
    obj = _GLOBAL_STAGE2[fam]
    return obj["tok"], obj["model"], obj["labels"]

def infer_partition_hier(rows_iter):
    dev = _device()
    stage1_tok, stage1_mdl, stage1_labels = _get_stage1()
    stage1_mdl.to(dev)
    stage1_mdl.eval()

    rows = list(rows_iter)
    if not rows:
        return

    payload = []
    for r in rows:
        rd = r.asDict(recursive=True)
        txt = (rd.get(CTA_COL) or "").strip()
        lob = (rd.get(LOB_COL) or "").strip()
        payload.append((rd, txt, lob))

    BATCH = 64
    families = []
    with torch.no_grad():
        for batch in _batch(payload, BATCH):
            texts = [x[1] if x[1] else "" for x in batch]
            enc = stage1_tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            logits = stage1_mdl(**enc).logits
            probs = _softmax(logits)
            top_ids = probs.argmax(dim=-1).tolist()
            families.extend([stage1_labels[i] for i in top_ids])

    by_family = {}
    for (rd, txt, lob), fam in zip(payload, families):
        by_family.setdefault(fam, []).append((rd, txt, lob))

    outputs = []
    with torch.no_grad():
        for fam, items in by_family.items():
            s2_tok, s2_mdl, s2_labels = _get_stage2(fam)
            s2_mdl.to(dev)
            s2_mdl.eval()
            for batch in _batch(items, BATCH):
                texts = [x[1] if x[1] else "" for x in batch]
                enc = s2_tok(texts, padding=True, truncation=True, max_length=64, return_tensors="pt")
                enc = {k: v.to(dev) for k, v in enc.items()}
                logits = s2_mdl(**enc).logits
                probs = _softmax(logits)
                top_ids = probs.argmax(dim=-1).tolist()
                top_labels = [s2_labels[i] for i in top_ids]

                for (rd, txt, lob), fine_intent in zip(batch, top_labels):
                    rd["intent_label"] = fine_intent
                    rd["canonical_cta"] = canonicalize_cta(txt, lob, fine_intent)
                    rd["category"] = derive_ms_category(lob, fine_intent, txt)
                    rd["coarse_family"] = fam
                    outputs.append(Row(**rd))
    for o in outputs:
        yield o

all_lobs = [row[LOB_COL] for row in df.select(LOB_COL).distinct().collect()]
priority_lobs = ["Windows", "Surface"]
other_lobs = [lob for lob in all_lobs if lob not in priority_lobs and lob is not None]
lobs_to_process = priority_lobs + other_lobs

enriched_schema = StructType([
    *df.schema,
    StructField("intent_label", StringType(), True),
    StructField("canonical_cta", StringType(), True),
    StructField("category", StringType(), True),
    StructField("coarse_family", StringType(), True),
])

for lob_name in lobs_to_process:
    if not lob_name:
        continue
    lob_df = df.filter(F.lower(F.col(LOB_COL)).contains(lob_name.lower()))
    if lob_df.limit(1).count() == 0:
        continue
    enriched_rdd = lob_df.rdd.mapPartitions(infer_partition_hier)
    enriched_df = spark.createDataFrame(enriched_rdd, schema=enriched_schema)
    out_path_lob = f"{OUT_PATH}/{lob_name}"
    (enriched_df
        .coalesce(1)
        .write
        .mode("append")
        .option("header", "true")
        .csv(out_path_lob))
    print(f"Finished processing {lob_name}, saved to {out_path_lob}")
