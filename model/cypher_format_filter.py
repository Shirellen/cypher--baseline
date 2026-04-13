import re
from typing import Dict, Optional, Set, Tuple, List, Any

def _cypher_tokens(cypher: str) -> Tuple[Set[str], Set[Tuple[str, str]], Dict[str, str]]:
    """
    参考 _sql_tokens 函数的逻辑来解析 Cypher 查询
    提取变量、属性引用和变量到标签的映射
    
    Returns:
        tokens: 所有标识符
        dotted: (变量, 属性) 对的集合
        alias_map: 变量到标签的映射 {变量: 标签}
    """
    # 清理查询字符串
    s = cypher.replace("`", " ").replace('"', " ").replace("'", " ")
    
    # 1. 提取所有标识符
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", s))
    
    # 2. 提取 variable.property 形式
    dotted = set()
    for match in re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z_][a-zA-Z0-9_]*)", s):
        dotted.add((match[0], match[1]))
    
    # 3. 提取 MATCH 子句中的变量和标签映射
    alias_map = {}
    
    # 查找所有 MATCH 模式中的变量:标签定义
    # 支持多种格式：(var:Label), (var:Label1:Label2), (var)
    match_patterns = re.findall(r"match\s+.*?(?=where|return|with|limit|$)", s, re.IGNORECASE | re.DOTALL)
    
    for pattern in match_patterns:
        # 查找节点定义：(variable:Label) 或 (variable:Label1:Label2)
        node_matches = re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)(?::[a-zA-Z_][a-zA-Z0-9_]*)?\)", pattern)
        for var, label in node_matches:
            alias_map[var.lower()] = label
        
        # 查找关系模式中的节点：()-[rel:TYPE]->(var:Label)
        rel_node_matches = re.findall(r"->\s*\(([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)(?::[a-zA-Z_][a-zA-Z0-9_]*)?\)", pattern)
        for var, label in rel_node_matches:
            alias_map[var.lower()] = label
        
        # 查找关系模式开头的节点：(var:Label)-[rel:TYPE]->
        start_node_matches = re.findall(r"\(([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z_][a-zA-Z0-9_]*)(?::[a-zA-Z_][a-zA-Z0-9_]*)?\)\s*-", pattern)
        for var, label in start_node_matches:
            alias_map[var.lower()] = label
    
    return tokens, dotted, alias_map

def _clean_filter_value(value: str) -> str:
    """清理过滤条件的值"""
    # 移除参数标记 $autostring_1 -> 实际值需要从查询参数中获取
    # 这里简化处理，保留数字值，参数值设为占位符
    value = value.strip()
    
    # 如果是参数，尝试提取数字
    if value.startswith('$'):
        # 对于 $autoint_0, $autostring_1 等，可以设置默认值
        if 'int' in value:
            return "0"  # 或者从查询上下文中获取实际值
        elif 'string' in value:
            return "'unknown'"
        else:
            return "0"
    
    # 移除引号
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    
    return value

def _parse_filter_details_flexible(details: str, original_query: str = None) -> Tuple[List[str], Optional[str]]:
    """
    灵活解析 Filter 操作符的 Details
    
    Args:
        details: Filter 操作的 Details 字段
        original_query: 原始 Cypher 查询字符串
    
    Returns:
        (过滤条件列表, 标签别名)
    """
    filters = []   
    # 分割多个条件
    conditions = re.split(r'\s+AND\s+|\s+OR\s+', details, flags=re.IGNORECASE)
    for condition in conditions:
        condition = condition.strip()
        # 2. 属性过滤条件：p.gender = value
        prop_match = re.search(r'([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\s*(=|<|>|<=|>=|<>)\s*(.+)', condition)
        if prop_match:
            var, prop, op, value = prop_match.groups()
            clean_value = _clean_filter_value(value)
            filters.append(f"{prop} {op} {clean_value}")
        
        # 3. 缓存属性过滤：cache[f.creationDate] > value
        cache_match = re.search(r'cache\[([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\]\s*(=|<|>|<=|>=|<>)\s*(.+)', condition)
        if cache_match:
            var, prop, op, value = cache_match.groups()
            clean_value = _clean_filter_value(value)
            filters.append(f"{prop} {op} {clean_value}")
    return filters

def extract_query_parameters(cypher_query: str) -> Dict[str, Any]:
    """
    从 Cypher 查询字符串中提取参数值
    """
    params = {}
    
    # 1. 提取数字比较条件
    number_patterns = [
        (r'creationDate\s*>\s*(\d+)', 'autoint_0'),
        (r'year\s*=\s*(\d+)', 'autoint_1'),
        (r'id\s*<\s*(\d+)', 'autoint_2'),
    ]
    
    for pattern, param_name in number_patterns:
        match = re.search(pattern, cypher_query, re.IGNORECASE)
        if match:
            params[param_name] = int(match.group(1))
    
    # 2. 提取字符串比较条件
    string_patterns = [
        (r"gender\s*=\s*'([^']+)'", 'autostring_1'),
        (r"name\s*=\s*'([^']+)'", 'autostring_2'),
        (r"title\s*CONTAINS\s*'([^']+)'", 'autostring_3'),
    ]
    
    for pattern, param_name in string_patterns:
        match = re.search(pattern, cypher_query, re.IGNORECASE)
        if match:
            params[param_name] = match.group(1)
    
    return params

def resolve_cypher_parameters(details: str, query_params: Dict[str, Any], original_query: str = "") -> str:
    """
    将 Details 中的参数占位符替换为实际值
    """
    resolved = details
    
    # 替换参数占位符
    for param_name, value in query_params.items():
        placeholder = f'${param_name}'
        if isinstance(value, str):
            resolved = resolved.replace(placeholder, f"'{value}'")
        else:
            resolved = resolved.replace(placeholder, str(value))
    
    # 兜底：仍存在 $autostring_* 或 $autoint_*，尝试依据属性名在 original_query 中回填
    if original_query:
        # 收集 details 中出现的属性名（如 cache[c.name] -> name）
        props = re.findall(r"(?:cache\[)?[a-zA-Z_]\w*\.([a-zA-Z_]\w*)", resolved)
        props = list(dict.fromkeys(props))
        
        # 替换字符串参数
        for placeholder in set(re.findall(r"\$autostring_\d+", resolved)):
            filled = False
            for prop in props:
                m = re.search(rf"\b{prop}\b\s*(=|CONTAINS|STARTS\s+WITH|ENDS\s+WITH)\s*['\"]([^'\"]+)['\"]", original_query, re.IGNORECASE)
                if m:
                    resolved = resolved.replace(placeholder, f"'{m.group(2)}'")
                    filled = True
                    break
            if not filled:
                # map literal in node pattern, e.g., (:University {name: 'Stanford'})
                for prop in props:
                    m2 = re.search(rf"\{{\s*{prop}\s*:\s*['\"]([^'\"]+)['\"]\s*\}}", original_query, re.IGNORECASE)
                    if m2:
                        resolved = resolved.replace(placeholder, f"'{m2.group(1)}'")
                        filled = True
                        break
        
        # 替换数值参数
        for placeholder in set(re.findall(r"\$autoint_\d+", resolved)):
            filled = False
            for prop in props:
                m = re.search(rf"\b{prop}\b\s*(=|>=|<=|>|<)\s*(\d+)", original_query, re.IGNORECASE)
                if m:
                    resolved = resolved.replace(placeholder, m.group(2))
                    filled = True
                    break
    
    return resolved

# 更新主函数
def cypher_format_filter(plan_node, original_query: str = None) -> Tuple[List[str], Optional[str]]:
    """
    从 Cypher 计划节点提取过滤条件和别名
    
    Args:
        plan_node: Cypher 执行计划节点
        original_query: 原始 Cypher 查询字符串（用于灵活推断别名）
    """
    # 提取所有标签
    # 解析查询获取 alias_map
    _, _, alias_map = _cypher_tokens(original_query)

    operator_type = plan_node.get('operatorType', '')
    args = plan_node.get('args', {})
    details = args.get('Details', '')
    
    # 如果提供了原始查询，解析并替换参数
    query_params = {}
    if original_query:
        query_params = extract_query_parameters(original_query)
        details = resolve_cypher_parameters(details, query_params, original_query)
    
    filters = []
    alias = []
    
    # 使用灵活的解析函数
    if 'Filter' in operator_type and details:
        filters = _parse_filter_details_flexible(details, original_query)
        variables = _extract_variables_from_details(details)
        """从 alias_map 中解析变量对应的标签"""
        for var in variables:
            if var.lower() in alias_map:
                alias.append(alias_map[var.lower()])
        # 直接匹配 var:Label（如 anon_5:University 或 m:Person）
        for _var, _label in re.findall(r"([a-zA-Z_]\w*):([a-zA-Z_]\w*)", details):
            alias.append(_label)
    elif operator_type in ['NodeByLabelScan', 'PartitionedNodeByLabelScan'] and details:
        # 对于标签扫描，直接从 Details 中提取
        variables = _extract_variables_from_details(details)
        """从 alias_map 中解析变量对应的标签"""
        for var in variables:
            if var.lower() in alias_map:
                alias.append(alias_map[var.lower()])
    elif operator_type in ['IntersectionNodeByLabelsScan', 'PartitionedIntersectionNodeByLabelsScan', 'SubtractionNodeByLabelsScan','PartitionedSubtractionNodeByLabelsScan'] and details:
        match = re.search(r'([a-zA-Z_]\w*):([a-zA-Z_]\w*)', details)
        if match:
            var, label = match.groups()
            alias = label  # 使用标签名作为别名
    # 对alias去重
    alias = list(dict.fromkeys(alias))
    return filters, alias

def _extract_variables_from_details(details):
    """从 Details 中提取所有变量"""
    variables = []
    
    # 1. 提取属性引用中的变量：p.gender, f.title, cache[p.firstName]
    prop_vars = re.findall(r'(?:cache\[)?([a-zA-Z_]\w*)\.', details)
    variables.extend(prop_vars)
    
    # 2. 提取关系模式中的变量：(f)-[rel]->(p)
    relation_vars = re.findall(r'\(([a-zA-Z_]\w*)\)', details)
    variables.extend(relation_vars)
    
    # 3. 提取简单的变量引用（在某些上下文中）
    simple_vars = re.findall(r'\b([a-z])\b', details)  # 单字母变量
    variables.extend(simple_vars)
    # 去重并保持顺序
    return list(dict.fromkeys(variables))
