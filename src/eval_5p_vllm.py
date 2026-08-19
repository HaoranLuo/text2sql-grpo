#!/usr/bin/env python3
"""vLLM 版 5prompt 投票评估 —— src/eval_5prompt_agent.py 的 vLLM 等价实现（贪心 T=0）。

与 HF 版的关系：
1. 推理引擎换为 vLLM（enable_lora=True + LoRARequest），解码参数对齐 HF 贪心
   （temperature=0 / top_p=1 / top_k=-1 / seed 固定），目标：生成文本与 HF
   do_sample=False 逐字节一致（需用验证脚本实测，50/50 全等为硬门槛）。
2. 输入构造与 HF 版完全相同：同一个 tokenizer、同一 chat template、
   同一 max_length=1536 截断，直接以 prompt_token_ids 送入 vLLM，
   避免 vLLM 端 tokenizer/chat-template 行为差异。
3. items.json 顶层 match/votes/truncated/db_id/predicted_sql 与 HF 版同构
   （eval_official.sh 兼容）；另加 candidates（全部候选 + 执行结果）与
   question，供未来离线组合实验使用。

新增能力（相对 HF 版）：
A. 中央候选库集成（src/candidate_store.py）：
   - 生成前 query()：命中（>= n_prompts 条）的题跳过生成，直接用缓存候选投票；
   - 生成后 ingest() 把新生成的候选入库；
   - --no-cache 完全关闭读写（一致性验证用）。
B. 候选 SQL 去重：同题内按 strip+lower+空白折叠归一化，每个唯一 SQL 只执行一次；
   投票按逐候选计票（等价于原版逐条执行），去重信息保留在 dup_count/vote。
C. 执行并行化：SQL 执行用线程池（--exec-workers，默认 16）。
   DatabaseExecutor.execute() 每次调用自开 sqlite 连接、无共享可变状态（已核对
   源码，线程安全），仍按约定用 thread-local 实例防御。

用法:
    python src/eval_5p_vllm.py --lora-path checkpoints/sft_v2 \
        --output-dir outputs/eval_5p_vllm_sftv2 --limit 10 --max-new-tokens 2048
"""
import argparse
import hashlib
import json
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 离线模式：避免 vLLM/transformers 尝试访问 HF Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from transformers import AutoTokenizer

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from eval_5prompt_agent import build_prompt_variants  # 与 HF 版共享同一 prompt 构造
from reasoning_generator_agent import ReasoningGeneratorAgent
from spider_utils import SpiderLoader, DatabaseExecutor, compare_execution_results

try:
    from candidate_store import ingest, query
    _STORE_AVAILABLE = True
except ImportError:
    _STORE_AVAILABLE = False

BASE_MODEL = str(PROJECT / 'models' / 'Qwen2.5-Coder-3B-Instruct')  # 默认，可用 --base-model 覆盖
SPIDER = str(PROJECT / 'data' / 'spider_data')
MODEL_NAME = "qwen2.5-coder-3b"  # 候选库中的模型名约定


def _norm(sql: str) -> str:
    """SQL 文本归一化（去重键）：strip + lower + 空白折叠。"""
    return " ".join(sql.strip().lower().split())


def result_rows_hash(full_rows) -> str:
    """执行结果行的稳定哈希（与候选库 result_hash 同源同口径）。"""
    payload = json.dumps(full_rows, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()[:16]


# thread-local DatabaseExecutor：防御性隔离（execute() 本身每次自开连接，
# 已核对无共享可变状态，但按约定逐线程建实例）
_tls = threading.local()


def _thread_executor() -> DatabaseExecutor:
    ex = getattr(_tls, "db_executor", None)
    if ex is None:
        ex = DatabaseExecutor(SPIDER)
        _tls.db_executor = ex
    return ex


def _exec_sql(task):
    db_id, sql = task
    return db_id, _norm(sql), _thread_executor().execute(db_id, sql)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lora-path', default=None, help='LoRA adapter 目录（省略=无 LoRA 基线）')
    parser.add_argument('--output-dir', required=True, help='输出目录')
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--n-prompts', type=int, default=5, choices=[5, 7],
                        help='投票用多少个 prompt 视角 (5 或 7)')
    parser.add_argument('--base-model', default=BASE_MODEL,
                        help='基础模型路径（默认 3B）')
    parser.add_argument('--no-cache', action='store_true',
                        help='关闭候选库读写（贪心一致性验证必须加此开关）')
    parser.add_argument('--exec-workers', type=int, default=16,
                        help='SQL 执行线程池大小（8-16）')
    parser.add_argument('--lora-name', default=None,
                        help='候选库中的 lora 名（默认取 --lora-path 的目录名）')
    parser.add_argument('--model-name', default=MODEL_NAME,
                        help='候选库中的模型名')
    parser.add_argument('--enforce-eager', action='store_true',
                        help='vLLM 关闭 flash-attn/CUDA graph（一致性重试用）')
    parser.add_argument('--attn-backend', default=None,
                        help='vLLM 注意力后端（FLASH_ATTN/TORCH_SDPA/XFORMERS，一致性重试用）')
    parser.add_argument('--max-num-seqs', type=int, default=None,
                        help='vLLM 单批最大序列数（=1 时逐条处理，对齐 HF batch=1 数值）')
    parser.add_argument('--kv-cache-dtype', default=None,
                        help='vLLM KV cache dtype（auto/fp16/bfloat16）')
    args = parser.parse_args()

    lora = args.lora_path
    out_dir = Path(args.output_dir)
    limit = args.limit
    start_index = args.start_index
    n_prompts = args.n_prompts
    base_model = args.base_model
    use_cache = (not args.no_cache)
    lora_name = args.lora_name or (Path(lora).name if lora else None)
    model_name = args.model_name
    print(f"{n_prompts}prompt投票[vLLM] | LoRA: {lora} | base: {base_model} | "
          f"{limit} 条 (start={start_index}) | T=0 贪心")
    print(f"候选库: {'启用' if use_cache else '关闭'} | "
          f"model={model_name} lora={lora_name} temperature=0.0 seed=0")

    # ---- 与 HF 版完全相同的 tokenizer 与输入构造 ----
    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True,
                                              trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    loader = SpiderLoader(SPIDER)
    items = loader.load_dev(limit=limit, start_index=start_index)

    # 先构造全部 (item, prompts, input_ids)，按题缓存命中情况分批送入 vLLM
    per_item_inputs = []  # (item, prompts, [ids_view0..ids_view4])
    for item in items:
        db_id, question = item['db_id'], item['question']
        ddl, _ = loader.get_ddl_with_source(db_id)
        prompts = build_prompt_variants(question, ddl)[:n_prompts]
        id_lists = []
        for p in prompts:
            chat = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': p}], tokenize=False,
                add_generation_prompt=True)
            ids = tokenizer(chat, truncation=True, max_length=1536)['input_ids']
            id_lists.append(ids)
        per_item_inputs.append((item, prompts, id_lists))

    # ---- 候选库查询：命中（>= n_prompts 条）的题跳过生成 ----
    # 注意：ingest 按 (配置, di, norm_sql, prompt_hash) 存储，prompt_hash 非空；
    # query 用 prompt_hash=None 只匹配空 hash 行，因此必须逐 prompt 按 hash 查询。
    cache_hits = {}  # di -> [候选 dict]
    gen_list = []    # (item, prompts, id_lists) 需要生成的题
    if use_cache:
        if not _STORE_AVAILABLE:
            print("WARNING: candidate_store 不可用，本次按无缓存运行")
            gen_list = per_item_inputs
        else:
            from candidate_store import prompt_hash as _prompt_hash
            for item, prompts, id_lists in per_item_inputs:
                di = item['dataset_index']
                hits_per_prompt = []
                all_hit = True
                for pi, p in enumerate(prompts):
                    h = query(di, model=model_name, lora=lora_name,
                              temperature=0.0, seed=0,
                              prompt_hash=_prompt_hash(p))
                    if len(h) >= 1:
                        hits_per_prompt.append(h[0])
                    else:
                        all_hit = False
                        break
                if all_hit and len(hits_per_prompt) >= n_prompts:
                    cache_hits[di] = hits_per_prompt[:n_prompts]
                else:
                    gen_list.append((item, prompts, id_lists))
            print(f"候选库命中 {len(cache_hits)}/{len(per_item_inputs)} 题，"
                  f"需生成 {len(gen_list)} 题")
    else:
        gen_list = per_item_inputs

    # ---- vLLM 引擎 + LoRA ----
    out_by_prompt = {}
    generation_seconds = 0.0
    if gen_list:
        # 注意力后端必须在 import vllm 之前通过环境变量设定
        if args.attn_backend:
            os.environ["VLLM_ATTENTION_BACKEND"] = args.attn_backend
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
        import vllm
        print(f"vLLM version: {vllm.__version__} "
              f"(attn_backend={os.environ.get('VLLM_ATTENTION_BACKEND', 'default')})")

        t_init = time.time()
        llm_kwargs = dict(model=base_model, enable_lora=True, max_loras=1,
                          max_lora_rank=32, dtype='bfloat16', trust_remote_code=True,
                          seed=0, max_model_len=1536 + args.max_new_tokens)
        if args.enforce_eager:
            llm_kwargs['enforce_eager'] = True
        if args.max_num_seqs is not None:
            llm_kwargs['max_num_seqs'] = args.max_num_seqs
        if args.kv_cache_dtype is not None:
            llm_kwargs['kv_cache_dtype'] = args.kv_cache_dtype
        llm = LLM(**llm_kwargs)
        lora_req = None
        if lora:
            lora_req = LoRARequest('sft_v2', 1, lora)
        print(f"vLLM engine ready in {time.time() - t_init:.1f}s")

        # T=0 贪心：temperature=0 强制 argmax，top_k=-1 关闭 top-k，固定 seed
        sampling_params = SamplingParams(temperature=0.0, top_p=1.0, top_k=-1,
                                         max_tokens=args.max_new_tokens, seed=0)

        all_ids = [ids for _, _, lst in gen_list for ids in lst]
        print(f"Generating {len(all_ids)} requests greedily...")
        t_gen = time.time()
        outputs = llm.generate([{'prompt_token_ids': ids} for ids in all_ids],
                               sampling_params, lora_request=lora_req)
        generation_seconds = time.time() - t_gen
        # vLLM 返回顺序不保证与输入一致 → 按 prompt_token_ids 匹配
        out_by_prompt = {tuple(o.prompt_token_ids): o for o in outputs}
        if len(out_by_prompt) != len(all_ids):
            print(f"WARNING: matched {len(out_by_prompt)}/{len(all_ids)} outputs by prompt ids")
        print(f"Generation done in {generation_seconds:.1f}s "
              f"({len(all_ids) / max(generation_seconds, 1e-9):.2f} req/s)")

    start_t = time.time()

    # ---- 组装每题的候选 ----
    # 生成路径候选: exec_ok=None 表示"待执行"；缓存路径: exec_ok=None 表示"未知→需执行"
    questions = []  # (item, candidates)
    for item, prompts, id_lists in per_item_inputs:
        di = item['dataset_index']
        if di in cache_hits:
            hits = cache_hits[di]
            cands = []
            for pi, h in enumerate(hits):
                cands.append({
                    'prompt_index': pi,
                    'raw': None,  # 缓存不含原始生成文本
                    'sql': (h.get('sql') or '').strip(),
                    'prompt': None,  # 缓存不存 prompt 原文（只存 hash）
                    'exec_ok': h.get('exec_ok'),  # True/False/None
                    'result_hash': h.get('result_hash'),
                    'vote': h.get('vote') if h.get('vote') is not None else 0,
                    'dup_count': 1,
                    'from_cache': True,
                })
            questions.append((item, cands))
            continue
        cands = []
        for pi, ids in enumerate(id_lists):
            o = out_by_prompt[tuple(ids)]
            text = tokenizer.decode(o.outputs[0].token_ids, skip_special_tokens=True)
            parsed = ReasoningGeneratorAgent.extract_sql(text)
            cands.append({
                'prompt_index': pi,
                'raw': text,
                'sql': parsed['sql'] if parsed['parse_success'] else '',
                'prompt': prompts[pi],
                'exec_ok': None,
                'result_hash': None,
                'vote': 0,
                'dup_count': 0,
                'from_cache': False,
            })
        # 同题内 SQL 归一化去重：dup_count = 该唯一 SQL 的出现次数
        norms = [_norm(c['sql']) for c in cands if c['sql']]
        counter = Counter(norms)
        for c in cands:
            if c['sql']:
                c['dup_count'] = counter[_norm(c['sql'])]
        questions.append((item, cands))

    # ---- 执行任务规划（按 (db_id, norm_sql) 去重）----
    exec_tasks = []  # (db_id, sql) 唯一执行任务
    exec_keys = set()
    total_candidate_sqls = 0
    for item, cands in questions:
        db_id = item['db_id']
        for c in cands:
            if not c['sql']:
                continue
            total_candidate_sqls += 1
            key = (db_id, _norm(c['sql']))
            if c['from_cache'] and c['exec_ok'] is not None:
                continue  # 缓存已有执行结论，不重执行
            if key not in exec_keys:
                exec_keys.add(key)
                exec_tasks.append((db_id, c['sql']))
        gold_key = (db_id, _norm(item['query']))
        if gold_key not in exec_keys:
            exec_keys.add(gold_key)
            exec_tasks.append((db_id, item['query']))  # gold 一并并行执行
    print(f"执行规划: 候选 SQL {total_candidate_sqls} 条 → 去重后执行 {len(exec_tasks)} 条（含 gold）")

    t_exec = time.time()
    exec_map = {}
    if exec_tasks:
        with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
            for db_id, norm, r in pool.map(_exec_sql, exec_tasks):
                exec_map[(db_id, norm)] = r
    exec_seconds = time.time() - t_exec
    print(f"并行执行完成 in {exec_seconds:.1f}s ({args.exec_workers} workers)")

    # ---- 投票 / 判定（顶层逻辑与 HF 版逐行一致）----
    match_count = 0
    results = []
    for i, (item, cands) in enumerate(questions):
        db_id, question, gold_sql = item['db_id'], item['question'], item['query']
        di = item['dataset_index']

        if cands and cands[0]['from_cache']:
            # 缓存路径：未知执行结论的先补执行；按 result_hash 分组计票
            for c in cands:
                if c['sql'] and c['exec_ok'] is None:
                    r = exec_map[(db_id, _norm(c['sql']))]
                    if r['success']:
                        c['exec_ok'] = True
                        c['result_hash'] = result_rows_hash(r['full_rows'])
                    else:
                        c['exec_ok'] = False
            groups = {}
            for c in cands:
                if c['exec_ok'] and c['result_hash']:
                    g = groups.setdefault(c['result_hash'],
                                          {"count": 0, "shortest_sql": c['sql']})
                    g["count"] += 1
                    if len(c['sql']) < len(g["shortest_sql"]):
                        g["shortest_sql"] = c['sql']
            voted_rows, vc, voted_truncated, selected_sql = [], 0, False, ""
            if groups:
                best = max(groups.values(), key=lambda g: g["count"])
                vc = best["count"]
                selected_sql = best["shortest_sql"]
                for c in cands:
                    if c['exec_ok'] and c['result_hash']:
                        c['vote'] = groups[c['result_hash']]["count"]
                # 缓存不含完整行集：重执行胜者 SQL 补回 voted_rows/truncated
                r = exec_map.get((db_id, _norm(selected_sql)))
                if r is None:
                    r = _thread_executor().execute(db_id, selected_sql)
                if r['success']:
                    voted_rows = [list(row) for row in r['full_rows']]
                    voted_truncated = r['full_rows_truncated']
        else:
            # 生成路径：与原版完全一致的执行结果分组投票
            exec_results = []  # (rows_tuple, truncated_flag, sql) —— 仅执行成功者
            for c in cands:
                if not c['sql']:
                    c['exec_ok'] = False
                    continue
                r = exec_map[(db_id, _norm(c['sql']))]
                if r['success']:
                    rows = tuple(tuple(row) for row in r['full_rows'])
                    c['exec_ok'] = True
                    c['result_hash'] = result_rows_hash(r['full_rows'])
                    exec_results.append((rows, r['full_rows_truncated'], c['sql']))
                else:
                    c['exec_ok'] = False

            voted_rows, vc, voted_truncated, selected_sql = [], 0, False, ""
            if exec_results:
                groups = {}
                for rows, truncated, sql in exec_results:
                    g = groups.setdefault(rows, {"count": 0, "truncated": False,
                                                 "shortest_sql": sql})
                    g["count"] += 1
                    g["truncated"] = g["truncated"] or truncated
                    if len(sql) < len(g["shortest_sql"]):
                        g["shortest_sql"] = sql
                best = max(groups.values(), key=lambda g: g["count"])
                voted_rows = [list(row) for row in
                              next(k for k, v in groups.items() if v is best)]
                vc = best["count"]
                voted_truncated = best["truncated"]
                selected_sql = best["shortest_sql"]
                group_counts = {result_rows_hash(rows): g["count"]
                                for rows, g in groups.items()}
                for c in cands:
                    if c['exec_ok']:
                        c['vote'] = group_counts.get(c['result_hash'], 0)

        gold_r = exec_map[(db_id, _norm(gold_sql))]
        gold_rows = gold_r['full_rows'] if gold_r['success'] else []
        gold_truncated = gold_r.get('full_rows_truncated', False) if gold_r['success'] else False

        if gold_r['success'] and not voted_truncated and not gold_truncated:
            is_match = compare_execution_results(
                voted_rows, gold_rows, gold_sql=gold_sql)['match']
        else:
            is_match = False
        if is_match:
            match_count += 1

        results.append({'di': di, 'match': is_match,
                        'votes': vc, 'truncated': voted_truncated,
                        'db_id': db_id,
                        'predicted_sql': selected_sql,
                        'question': question,
                        'candidates': cands})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{limit}] match={match_count}/{i+1} ({match_count/(i+1):.1%})")

    elapsed = time.time() - start_t
    rate = match_count / limit
    print(f"\n=== 5prompt投票 RESULT [vLLM] ===")
    print(f"Match: {match_count}/{limit} ({rate:.1%})")
    print(f"Time: {elapsed:.0f}s (generation {generation_seconds:.1f}s, "
          f"exec {exec_seconds:.1f}s)")
    print(f"LoRA: {lora}")

    # 候选库写入：仅写入本次新生成的题（缓存命中的题已在库中）
    if use_cache and _STORE_AVAILABLE:
        generated_results = [r for r in results
                             if r['candidates'] and not r['candidates'][0]['from_cache']]
        if generated_results:
            res = ingest(generated_results, model=model_name,
                         lora=lora_name, temperature=0.0, seed=0)
            print(f"候选库 ingest: +{res['added']} (跳过 {res['skipped']})")

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'method': '5prompt_vote', 'engine': 'vllm',
                   'lora': lora,
                   'match_rate': rate, 'match_count': match_count,
                   'start_index': start_index, 'limit': limit,
                   'elapsed_seconds': round(elapsed, 1),
                   'generation_seconds': round(generation_seconds, 1),
                   'exec_seconds': round(exec_seconds, 1),
                   'max_new_tokens': args.max_new_tokens,
                   'n_prompts': n_prompts,
                   'cache': {'enabled': use_cache,
                             'hits': len(cache_hits),
                             'generated': len(gen_list)},
                   'dedup': {'total_candidate_sqls': total_candidate_sqls,
                             'unique_executed': len(exec_tasks)},
                   'exec_workers': args.exec_workers,
                   'model_name': model_name,
                   'lora_name': lora_name}, f, indent=2)
    with open(out_dir / 'items.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
