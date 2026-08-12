################################
# Assumptions:
#   1. sql is correct
#   2. only table name has alias
#   3. only one intersect/union/except
#
# val: number(float)/string(str)/sql(dict)
# col_unit: (agg_id, col_id, isDistinct(bool))
# val_unit: (unit_op, col_unit1, col_unit2)
# table_unit: (table_type, col_unit/sql)
# cond_unit: (not_op, op_id, val_unit, val1, val2)
# condition: [cond_unit1, 'and'/'or', cond_unit2, ...]
# sql {
#   'select': (isDistinct(bool), [(agg_id, val_unit), (agg_id, val_unit), ...])
#   'from': {'table_units': [table_unit1, table_unit2, ...], 'conds': condition}
#   'where': condition
#   'groupBy': [col_unit1, col_unit2, ...]
#   'orderBy': ('asc'/'desc', [val_unit1, val_unit2, ...])
#   'having': condition
#   'limit': None/limit value
#   'intersect': None/sql
#   'except': None/sql
#   'union': None/sql
# }
################################

import json
import sqlite3
from nltk import word_tokenize

CLAUSE_KEYWORDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
JOIN_KEYWORDS = ('join', 'on', 'as')

WHERE_OPS = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
UNIT_OPS = ('none', '-', '+', "*", '/')
AGG_OPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TABLE_TYPE = {
    'sql': "sql",
    'table_unit': "table_unit",
}

COND_OPS = ('and', 'or')
SQL_OPS = ('intersect', 'union', 'except')
ORDER_OPS = ('desc', 'asc')



class Schema:
    """
    Simple schema which maps table&column to a unique identifier
    """
    def __init__(self, schema):
        self._schema = schema
        self._idMap = self._map(self._schema)

    @property
    def schema(self):
        return self._schema

    @property
    def idMap(self):
        return self._idMap

    def _map(self, schema):
        idMap = {'*': "__all__"}
        id = 1
        for key, vals in schema.items():
            for val in vals:
                idMap[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1

        for key in schema:
            idMap[key.lower()] = "__" + key.lower() + "__"
            id += 1

        return idMap


def get_schema(db):
    """
    Get database's schema, which is a dict with table name as key
    and list of column names as value
    :param db: database path
    :return: schema dict
    """

    schema = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    # fetch table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]

    # fetch table info
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        schema[table] = [str(col[1].lower()) for col in cursor.fetchall()]

    return schema


def get_schema_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)

    schema = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        schema[table] = cols

    return schema


def tokenize(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_idxs = [idx for idx, char in enumerate(string) if char == '"']
    assert len(quote_idxs) % 2 == 0, "Unexpected quote"

    # keep string value as token
    vals = {}
    for i in range(len(quote_idxs)-1, -1, -2):
        qidx1 = quote_idxs[i-1]
        qidx2 = quote_idxs[i]
        val = string[qidx1: qidx2+1]
        key = "__val_{}_{}__".format(qidx1, qidx2)
        string = string[:qidx1] + key + string[qidx2+1:]
        vals[key] = val

    # FIX: 强制拆开 '='（nltk 对 name=value 无空格粘连不切分，
    # 产出单 token 'name=__val_...__' 导致 'Error col: name=__val_104_109__'；
    # 既有第 139-146 行合并逻辑会把 !=/>=/<= 还原）
    string = string.replace('=', ' = ')

    toks = [word.lower() for word in word_tokenize(string)]
    # replace with string value token
    for i in range(len(toks)):
        if toks[i] in vals:
            toks[i] = vals[toks[i]]

    # find if there exists !=, >=, <=
    eq_idxs = [idx for idx, tok in enumerate(toks) if tok == "="]
    eq_idxs.reverse()
    prefix = ('!', '>', '<')
    for eq_idx in eq_idxs:
        pre_tok = toks[eq_idx-1]
        if pre_tok in prefix:
            toks = toks[:eq_idx-1] + [pre_tok + "="] + toks[eq_idx+1: ]

    # FIX: 合并 '<>' 为 '!='（nltk 把 '<>' 拆成 '<','>' 两个 token，
    # '>' 会被误当列名解析报 'Error col: >'；WHERE_OPS 已含 '!=' 与 gold 对齐）
    new_toks = []
    i = 0
    while i < len(toks):
        if i + 1 < len(toks) and toks[i] == '<' and toks[i+1] == '>':
            new_toks.append('!=')
            i += 2
        else:
            new_toks.append(toks[i])
            i += 1
    toks = new_toks

    # FIX: 粘连 token 拆分（nltk 对无空格粘连片段不切分，如 IN (9,10) 的 '9,10' 单 token；
    # 带引号的字符串值不被拆分）
    new_toks = []
    for tok in toks:
        if ',' in tok and not tok.startswith('"'):
            new_toks.extend([part for part in tok.split(',') if part != ''])
        else:
            new_toks.append(tok)
    toks = new_toks

    return toks


def scan_alias(toks):
    """Scan the index of 'as' and build the map for all alias"""
    as_idxs = [idx for idx, tok in enumerate(toks) if tok == 'as']
    alias = {}
    for idx in as_idxs:
        alias[toks[idx+1]] = toks[idx-1]
    return alias


def get_tables_with_alias(schema, toks):
    tables = scan_alias(toks)
    for key in schema:
        # FIX: 别名与表名同名（大小写不敏感，FROM airlines AS Airlines）时
        # 跳过映射而非断言崩溃——scan_alias 已把该别名映射到自身表名，与默认映射等价
        if key in tables:
            continue
        tables[key] = key
    return tables


def parse_col(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, column id
    """
    tok = toks[start_idx]
    if tok == "*":
        return start_idx + 1, schema.idMap[tok]

    # FIX: 常量字面量（字符串/数字，如 EXISTS 子查询的 SELECT 1 中的 '1'）当作伪列 id -1，
    # 与 eval 层伪列约定一致——该列跳过值比较，只影响分项不得分，不会崩溃
    if tok.startswith('"'):
        return start_idx + 1, -1
    try:
        float(tok)
        return start_idx + 1, -1
    except ValueError:
        pass

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        # FIX: 别名未知或派生表列（如 x.a）不在 schema 中时返回伪列 -1，避免 KeyError
        if alias in tables_with_alias:
            key = tables_with_alias[alias] + "." + col
            if key in schema.idMap:
                return start_idx+1, schema.idMap[key]
        return start_idx + 1, -1

    if default_tables is None or len(default_tables) == 0:
        # FIX: 无 FROM（SELECT 1）/派生表未注册列时回退遍历全部表，仍找不到返回伪列 -1
        for table in schema.schema:
            if tok in schema.schema[table]:
                key = table + "." + tok
                if key in schema.idMap:
                    return start_idx+1, schema.idMap[key]
        return start_idx + 1, -1

    for alias in default_tables:
        table = tables_with_alias[alias]
        if tok in schema.schema.get(table, []):
            key = table + "." + tok
            return start_idx+1, schema.idMap[key]

    # FIX: 列未命中 schema（ORDER BY 引用 SELECT 别名、模型臆造列名）时
    # 返回伪列 -1 而非断言崩溃 'Error col: ...'
    return start_idx + 1, -1


def parse_col_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    """
        :returns next idx, (agg_op id, col_id)
    """
    idx = start_idx
    len_ = len(toks)
    isBlock = False
    isDistinct = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # FIX: 聚合关键字后必须紧跟 '('；否则该 token（如 MAX(count) 中 count 是派生表别名）
    # 会被误判为聚合词并触发无消息空断言，需降级为普通列解析
    if toks[idx] in AGG_OPS and idx + 1 < len_ and toks[idx + 1] == '(':
        agg_id = AGG_OPS.index(toks[idx])
        idx += 1
        idx += 1  # skip '('
        if toks[idx] == "distinct":
            idx += 1
            isDistinct = True
        if toks[idx] == 'case':
            # FIX: SUM(CASE WHEN ... THEN ... END) —— CASE 表达式整体跳过（扫到匹配的 end），
            # 内部 when/op/val 无法按列解析
            case_depth = 0
            while idx < len_:
                if toks[idx] == 'case':
                    case_depth += 1
                elif toks[idx] == 'end':
                    case_depth -= 1
                    if case_depth == 0:
                        idx += 1
                        break
                idx += 1
            col_id = -1
        else:
            # FIX: 聚合参数可能是嵌套表达式（SUM(CAST(col AS type)) 等）——
            # 交给 parse_val_unit 完整解析后取列 id，避免 'cast' 被当列名解析后断言失败
            idx, inner_val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
            col_id = inner_val_unit[1][1]
        assert idx < len_ and toks[idx] == ')'
        idx += 1
        return idx, (agg_id, col_id, isDistinct)

    if toks[idx] == "distinct":
        idx += 1
        isDistinct = True
    agg_id = AGG_OPS.index("none")
    idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        # FIX: idx < len_ 越界保护（生成截断缺少右括号时避免 IndexError）
        assert idx < len_ and toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (agg_id, col_id, isDistinct)


def parse_val_unit(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    isBlock = False

    # FIX: 标量子查询作比较操作数 —— (SELECT ...) > N 整体当作伪列单元，
    # 由 parse_condition 按正常运算符路径处理（parse_value 只覆盖运算符右侧，覆盖不了左侧/HAVING）
    if idx < len_ and toks[idx] == '(' and idx + 1 < len_ and toks[idx + 1] == 'select':
        isBlock = True
        idx += 2
        try:
            idx, _sub = parse_sql(toks, idx, tables_with_alias, schema)
        except Exception:
            depth = 1
            while idx < len_ and depth > 0:
                if toks[idx] == '(':
                    depth += 1
                elif toks[idx] == ')':
                    depth -= 1
                idx += 1
        if idx < len_ and toks[idx] == ')':
            idx += 1
        return idx, (0, (-1, -1, False), None)  # 伪列单元 (unit_op, col_unit1, col_unit2)

    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # FIX: EXISTS (SELECT ...) —— 伪列单元（列 -1，eval 层跳过该列比较）
    if idx < len_ and toks[idx].lower() == 'exists':
        idx += 1
        if idx < len_ and toks[idx] == '(':
            idx += 1
        if idx < len_ and toks[idx] == 'select':
            try:
                idx, _sub = parse_sql(toks, idx, tables_with_alias, schema)
            except Exception:
                # FIX: 容错按括号深度跳到匹配右括号，不再错跳到内层子查询的 ')' 或停在错误位置
                depth = 1
                while idx < len_ and depth > 0:
                    if toks[idx] == '(':
                        depth += 1
                    elif toks[idx] == ')':
                        depth -= 1
                    idx += 1
        if idx < len_ and toks[idx] == ')':
            idx += 1
        return idx, (0, (-1, -1, False), None)  # 伪列单元 (unit_op, col_unit1, col_unit2)

    # FIX: CAST(col AS type) —— 当作普通列单元解析
    if idx < len_ and toks[idx].lower() == 'cast':
        idx += 1
        if idx < len_ and toks[idx] == '(':
            idx += 1
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        while idx < len_ and toks[idx] not in (',', ')', 'from', 'where',
                                               'group', 'order', 'having', 'limit'):
            idx += 1  # 跳过 AS type
        if idx < len_ and toks[idx] == ')':
            idx += 1
        col_unit1 = (0, col_id, False)  # none agg
        unit_op = UNIT_OPS.index('none')
        # FIX: 闭合外层括号（AVG(CAST(...)) 中 AVG 的 ')' 需在此消费，
        # 否则残留 ')' 在 parse_select 循环里被当列名报 'Error col: )'）
        if isBlock:
            if idx < len_ and toks[idx] == ')':
                idx += 1
        return idx, (unit_op, col_unit1, None)

    # FIX: CASE WHEN ... THEN ... ELSE ... END —— 整体当作伪列单元（扫描到匹配的 end）
    if idx < len_ and toks[idx] == 'case':
        case_depth = 0
        while idx < len_:
            if toks[idx] == 'case':
                case_depth += 1
            elif toks[idx] == 'end':
                case_depth -= 1
                if case_depth == 0:
                    idx += 1
                    break
            idx += 1
        if idx < len_ and toks[idx] == ')':
            idx += 1
        return idx, (0, (-1, -1, False), None)  # 伪列单元

    # FIX: 标量函数 LOWER/UPPER/ROUND/LENGTH 等 —— 仿 CAST 分支解析 func(col)，
    # 排除关键词/聚合词/运算符/括号，避免误伤正常路径
    if idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in AGG_OPS \
            and toks[idx] not in WHERE_OPS and toks[idx] not in JOIN_KEYWORDS \
            and toks[idx] not in UNIT_OPS and toks[idx] not in COND_OPS \
            and toks[idx] not in ORDER_OPS and toks[idx] != '(' \
            and idx + 1 < len_ and toks[idx + 1] == '(':
        idx += 1
        if toks[idx] == '(':
            idx += 1
        idx, col_id = parse_col(toks, idx, tables_with_alias, schema, default_tables)
        if idx < len_ and toks[idx] == ')':
            idx += 1
        col_unit1 = (0, col_id, False)  # none agg
        unit_op = UNIT_OPS.index('none')
        if isBlock:
            if idx < len_ and toks[idx] == ')':
                idx += 1
        return idx, (unit_op, col_unit1, None)

    col_unit1 = None
    col_unit2 = None
    unit_op = UNIT_OPS.index('none')

    idx, col_unit1 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
    if idx < len_ and toks[idx] in UNIT_OPS:
        unit_op = UNIT_OPS.index(toks[idx])
        idx += 1
        idx, col_unit2 = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)

    if isBlock:
        # FIX: idx < len_ 越界保护（生成截断缺少右括号时避免 IndexError）
        assert idx < len_ and toks[idx] == ')'
        idx += 1  # skip ')'

    return idx, (unit_op, col_unit1, col_unit2)


def parse_table_unit(toks, start_idx, tables_with_alias, schema):
    """
        :returns next idx, table id, table name
    """
    idx = start_idx
    len_ = len(toks)
    key = tables_with_alias[toks[idx]]

    if idx + 1 < len_ and toks[idx+1] == "as":
        idx += 3
    else:
        idx += 1
        # FIX: 隐式表别名（FROM singer s）——下一个 token 非关键词时视为别名。
        # 必须排除 JOIN_KEYWORDS：'join'/'on'/'as' 若被注册为别名并被跳过，
        # FROM t1 JOIN t2 ON t1.a=t2.b 循环会把 ON 条件的首个列 token 当表名 → KeyError 't1.a'
        if idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in JOIN_KEYWORDS \
                and toks[idx] not in (",", ")", ";"):
            tables_with_alias[toks[idx]] = key  # 注册别名 → 表名
            idx += 1

    return idx, schema.idMap[key], key


def parse_value(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    isBlock = False
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    if toks[idx] == 'select':
        idx, val = parse_sql(toks, idx, tables_with_alias, schema)
    elif "\"" in toks[idx]:  # token is a string value
        val = toks[idx]
        idx += 1
    elif toks[idx] == 'null':
        # FIX: IS NULL / IS NOT NULL —— 'null' 是关键字而非列名，否则报 'Error col: null'
        val = None
        idx += 1
    else:
        try:
            val = float(toks[idx])
            idx += 1
        except:
            end_idx = idx
            while end_idx < len_ and toks[end_idx] != ',' and toks[end_idx] != ')'\
                and toks[end_idx] != 'and' and toks[end_idx] not in CLAUSE_KEYWORDS and toks[end_idx] not in JOIN_KEYWORDS:
                    end_idx += 1

            idx, val = parse_col_unit(toks[start_idx: end_idx], 0, tables_with_alias, schema, default_tables)
            idx = end_idx

    if isBlock:
        # FIX: idx < len_ 越界保护（生成截断缺少右括号时避免 IndexError）
        assert idx < len_ and toks[idx] == ')'
        idx += 1

    return idx, val


def parse_condition(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)
    conds = []

    while idx < len_:
        # FIX: NOT 前置（NOT EXISTS / NOT IN / NOT LIKE）——先于 val_unit 解析
        not_op = False
        if toks[idx] == 'not':
            not_op = True
            idx += 1

        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)

        # FIX: EXISTS/CASE 伪列单元（col_unit1 == (-1,-1,False)）是完整条件，其后没有运算符
        # （可能紧跟 and/or/子句关键字/结尾），直接 append 条件并跳过 WHERE_OPS 断言；
        # 否则 NOT EXISTS 族在语句末尾必然 idx==len_ 触发 IndexError，
        # 后接 and/union 时必然 AssertionError（结构性根因：NOT EXISTS 之后不存在运算符）
        is_pseudo_unit = val_unit[0] == 0 and val_unit[1] == (-1, -1, False) and val_unit[2] is None
        if is_pseudo_unit and not (idx < len_ and toks[idx] in WHERE_OPS):
            conds.append((not_op, WHERE_OPS.index('exists'), val_unit, None, None))
            if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
                break
            if idx < len_ and toks[idx] in COND_OPS:
                conds.append(toks[idx])
                idx += 1  # skip and/or
            continue

        # FIX: 断言消息越界安全化——idx 到达列表末尾时消息不再取 toks[idx]（避免 IndexError）
        assert idx < len_ and toks[idx] in WHERE_OPS, \
            "Error condition: idx: {}, tok: {}".format(idx, toks[idx] if idx < len_ else "<EOF>")
        op_id = WHERE_OPS.index(toks[idx])
        idx += 1
        val1 = val2 = None
        # FIX: 列后 NOT（col NOT IN / NOT LIKE / NOT BETWEEN）——'not' 被误当运算符时
        # 翻转 not_op 并改指真正的运算符，否则报 'Error col: in'
        if op_id == WHERE_OPS.index('not') and idx < len_ and toks[idx] in ('in', 'like', 'between', 'is'):
            not_op = True
            op_id = WHERE_OPS.index(toks[idx])
            idx += 1
        # FIX: IS NOT NULL —— 'not' 属于 IS 的否定，消费后由 parse_value 的 null 分支处理
        if op_id == WHERE_OPS.index('is') and idx < len_ and toks[idx] == 'not':
            not_op = True
            idx += 1
        if op_id == WHERE_OPS.index('between'):  # between..and... special case: dual values
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            # FIX: idx < len_ 越界保护（生成截断缺少 and 时避免 IndexError）
            assert idx < len_ and toks[idx] == 'and'
            idx += 1
            idx, val2 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
        elif op_id == WHERE_OPS.index('in') and idx < len_ and toks[idx] == '(' \
                and idx + 1 < len_ and toks[idx + 1] != 'select':
            # FIX: IN (v1, v2, ...) 多值列表——循环收集全部值直到 ')'，
            # 否则 parse_value 遇 ',' 触发无消息空断言（parse_value 只支持单值）
            idx += 1  # skip '('
            val1 = []
            while idx < len_ and toks[idx] != ')':
                idx, val = parse_value(toks, idx, tables_with_alias, schema, default_tables)
                val1.append(val)
                if idx < len_ and toks[idx] == ',':
                    idx += 1  # skip ','
            if idx < len_ and toks[idx] == ')':
                idx += 1
            val2 = None
        else:  # normal case: single value
            idx, val1 = parse_value(toks, idx, tables_with_alias, schema, default_tables)
            val2 = None

        conds.append((not_op, op_id, val_unit, val1, val2))

        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";") or toks[idx] in JOIN_KEYWORDS):
            break

        if idx < len_ and toks[idx] in COND_OPS:
            conds.append(toks[idx])
            idx += 1  # skip and/or

    return idx, conds


def parse_select(toks, start_idx, tables_with_alias, schema, default_tables=None):
    idx = start_idx
    len_ = len(toks)

    assert toks[idx] == 'select', "'select' not found"
    idx += 1
    isDistinct = False
    if idx < len_ and toks[idx] == 'distinct':
        idx += 1
        isDistinct = True
    val_units = []

    while idx < len_ and toks[idx] not in CLAUSE_KEYWORDS:
        agg_id = AGG_OPS.index("none")
        if toks[idx] in AGG_OPS:
            agg_id = AGG_OPS.index(toks[idx])
            idx += 1
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append((agg_id, val_unit))
        # FIX: 跳过 AS 别名（模型生成的 SQL 常用 AS，官方解析器不支持）
        if idx < len_ and toks[idx].lower() == 'as':
            idx += 2  # skip 'as' + alias_name
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','

    return idx, (isDistinct, val_units)


def parse_from(toks, start_idx, tables_with_alias, schema):
    """
    Assume in the from clause, all table units are combined with join
    """
    # FIX: SELECT 1 等无 FROM 的 SQL——返回空 from（上层容错继续解析），
    # 不再断言 "'from' not found" 崩溃
    if 'from' not in toks[start_idx:]:
        return start_idx, [], [], []

    len_ = len(toks)
    idx = toks.index('from', start_idx) + 1
    default_tables = []
    table_units = []
    conds = []

    while idx < len_:
        # FIX: join 变体前缀/join 须在 '(' 与 'select' 分支之前跳过，
        # 使 JOIN (SELECT ...) 子查询进入 isBlock/select 分支，否则 'select' 被当表名报 KeyError
        if idx < len_ and toks[idx] in ('left', 'right', 'full', 'cross', 'inner', 'outer'):
            idx += 1  # skip join 变体前缀 (left/right/full/...)
        if idx < len_ and toks[idx] == 'join':
            idx += 1  # skip join

        isBlock = False
        if toks[idx] == '(':
            isBlock = True
            idx += 1

        if toks[idx] == 'select':
            idx, sql = parse_sql(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['sql'], sql))
        else:
            idx, table_unit, table_name = parse_table_unit(toks, idx, tables_with_alias, schema)
            table_units.append((TABLE_TYPE['table_unit'],table_unit))
            default_tables.append(table_name)

        if isBlock:
            # FIX: 子查询缺右括号（生成截断）时容错跳出，避免 assert 越界
            if idx >= len_ or toks[idx] != ')':
                break
            idx += 1
            # FIX: 派生表别名 `) AS t` / `) t`——注册别名并加入 default_tables，
            # 避免 'as'/别名被当作表名解析而 KeyError（外层引用派生列时经 parse_col 伪列兜底）
            if idx < len_ and toks[idx] == 'as':
                idx += 1
            if idx < len_ and toks[idx] not in CLAUSE_KEYWORDS and toks[idx] not in (',', ')', ';') \
                    and toks[idx] not in JOIN_KEYWORDS:
                tables_with_alias[toks[idx]] = toks[idx]
                default_tables.append(toks[idx])
                idx += 1
        if idx < len_ and toks[idx] == "on":
            idx += 1  # skip on
            idx, this_conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)
        if idx < len_ and (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
            break

    return idx, table_units, conds, default_tables


def parse_where(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'where':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_group_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    col_units = []

    if idx >= len_ or toks[idx] != 'group':
        return idx, col_units

    idx += 1
    # FIX: 'group' 后无 'by'（生成截断）时直接返回，避免 toks[idx] 越界
    if idx >= len_ or toks[idx] != 'by':
        return idx, col_units
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, col_unit = parse_col_unit(toks, idx, tables_with_alias, schema, default_tables)
        col_units.append(col_unit)
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, col_units


def parse_order_by(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)
    val_units = []
    order_type = 'asc' # default type is 'asc'

    if idx >= len_ or toks[idx] != 'order':
        return idx, val_units

    idx += 1
    # FIX: 'order' 后无 'by'（生成截断）时直接返回，避免 toks[idx] 越界
    if idx >= len_ or toks[idx] != 'by':
        return idx, val_units
    idx += 1

    while idx < len_ and not (toks[idx] in CLAUSE_KEYWORDS or toks[idx] in (")", ";")):
        idx, val_unit = parse_val_unit(toks, idx, tables_with_alias, schema, default_tables)
        val_units.append(val_unit)
        if idx < len_ and toks[idx] in ORDER_OPS:
            order_type = toks[idx]
            idx += 1
        if idx < len_ and toks[idx] == ',':
            idx += 1  # skip ','
        else:
            break

    return idx, (order_type, val_units)


def parse_having(toks, start_idx, tables_with_alias, schema, default_tables):
    idx = start_idx
    len_ = len(toks)

    if idx >= len_ or toks[idx] != 'having':
        return idx, []

    idx += 1
    idx, conds = parse_condition(toks, idx, tables_with_alias, schema, default_tables)
    return idx, conds


def parse_limit(toks, start_idx):
    idx = start_idx
    len_ = len(toks)

    if idx < len_ and toks[idx] == 'limit':
        idx += 2
        # make limit value can work, cannot assume put 1 as a fake limit number
        # FIX: 'limit' 后无数字（生成截断）时越界保护
        if idx - 1 >= len(toks) or type(toks[idx-1]) != int:
            return idx, 1

        return idx, int(toks[idx-1])

    return idx, None


def parse_sql(toks, start_idx, tables_with_alias, schema):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(toks)
    idx = start_idx

    sql = {}
    if toks[idx] == '(':
        isBlock = True
        idx += 1

    # parse from clause in order to get default tables
    from_end_idx, table_units, conds, default_tables = parse_from(toks, start_idx, tables_with_alias, schema)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    # select clause
    _, select_col_units = parse_select(toks, idx, tables_with_alias, schema, default_tables)
    idx = from_end_idx
    sql['select'] = select_col_units
    # where clause
    idx, where_conds = parse_where(toks, idx, tables_with_alias, schema, default_tables)
    sql['where'] = where_conds
    # group by clause
    idx, group_col_units = parse_group_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['groupBy'] = group_col_units
    # having clause
    idx, having_conds = parse_having(toks, idx, tables_with_alias, schema, default_tables)
    sql['having'] = having_conds
    # order by clause
    idx, order_col_units = parse_order_by(toks, idx, tables_with_alias, schema, default_tables)
    sql['orderBy'] = order_col_units
    # limit clause
    idx, limit_val = parse_limit(toks, idx)
    sql['limit'] = limit_val

    idx = skip_semicolon(toks, idx)
    if isBlock:
        # FIX: idx < len_ 越界保护（生成截断缺少右括号时避免 IndexError）
        assert idx < len_ and toks[idx] == ')'
        idx += 1  # skip ')'
    idx = skip_semicolon(toks, idx)

    # intersect/union/except clause
    for op in SQL_OPS:  # initialize IUE
        sql[op] = None
    if idx < len_ and toks[idx] in SQL_OPS:
        sql_op = toks[idx]
        idx += 1
        idx, IUE_sql = parse_sql(toks, idx, tables_with_alias, schema)
        sql[sql_op] = IUE_sql
    return idx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data


def get_sql(schema, query):
    toks = tokenize(query)
    tables_with_alias = get_tables_with_alias(schema.schema, toks)
    _, sql = parse_sql(toks, 0, tables_with_alias, schema)

    return sql


def skip_semicolon(toks, start_idx):
    idx = start_idx
    while idx < len(toks) and toks[idx] == ";":
        idx += 1
    return idx
