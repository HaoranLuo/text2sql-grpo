"""诊断原始 Spider evaluation.py 为何静默退出"""
import sys, traceback

sys.path.insert(0, "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/tools/original_spider_eval")

print("STEP1: import start", flush=True)
try:
    import evaluation
    print("STEP2: import OK", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("STEP3: files check", flush=True)
gold = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/gold.sql"
pred = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/outputs/official_original_vav/pred.sql"
db = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/database"
table = "/gpfs/work/aac/jiahuiwang24/reasoning_generator_3b/data/spider_data/tables.json"
print(f"gold lines: {len(open(gold).readlines())}", flush=True)
print(f"pred lines: {len(open(pred).readlines())}", flush=True)

print("STEP4: build kmaps", flush=True)
try:
    kmaps = evaluation.build_foreign_key_map_from_json(table)
    print(f"kmaps: {len(kmaps)} dbs", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)

print("STEP5: call evaluate", flush=True)
try:
    evaluation.evaluate(gold, pred, db, "exec", kmaps)
    print("STEP6: evaluate returned", flush=True)
except Exception:
    traceback.print_exc()
    sys.exit(1)
