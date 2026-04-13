# cypher_adapter.py
import re

def cypher_format_join(plan_node):
    """提取关系遍历信息（类型+方向）"""
    args = plan_node.get('args', {})
    details = args.get('Details', '')
    
    if 'Expand' in plan_node.get('operatorType', ''):
        rel_match = re.search(r':([^]]+)', details)
        if rel_match:
            rel_type = rel_match.group(1).strip()
            
            # 分析遍历模式
            if re.search(r'\([^)]*\)-\[[^\]]*\]->\([^)]*\)', details):
                return f"{rel_type}_OUT"
            elif re.search(r'\([^)]*\)<-\[[^\]]*\]-\([^)]*\)', details):  
                return f"{rel_type}_IN"
            else:
                return rel_type
    
    return None

# 测试结果：
# "(f)-[anon_0:HAS_MEMBER]->(p)" -> "HAS_MEMBER_OUT"
# "(p)<-[anon_0:HAS_MEMBER]-(f)" -> "HAS_MEMBER_IN"