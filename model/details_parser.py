# Details 解析器
import re
from typing import Dict, List, Tuple, Set

class DetailsParser:
    def __init__(self):
        # 预编译正则表达式
        self.label_pattern = re.compile(r'(\w+):(\w+)')  # u:University
        # self.expand_pattern = re.compile(r'\((\w+)\)<?-?\[(\w+):(\w+)\]->?\((\w+)\)')  # (u)<-[s:STUDY_AT]-(p2)
        self.expand_pattern = re.compile(r'\((\w+)\)(<-|\-)\[(\w+):(\w+)\](\->|\-)\((\w+)\)')
        self.property_pattern = re.compile(r'(\w+)\.(\w+)')  # p.gender, cache[p.firstName]
        self.cache_property_pattern = re.compile(r'cache\[(\w+)\.(\w+)\]')  # cache[p.firstName]
        
    def parse_details(self, operator_type: str, details: str) -> Dict:
        """根据操作符类型解析Details字符串"""
        result = {
            'nodes': set(),
            'relationships': set(), 
            'properties': set(),
            'node_properties': {},
            'rel_properties': {},
            'graph_structure': []
        }
        
        if not details:
            return result
            
        # 根据操作符类型分发解析
        if 'Scan' in operator_type:
            return self._parse_scan_details(details, result)
        elif 'Expand' in operator_type:
            return self._parse_expand_details(details, result)
        elif 'Filter' in operator_type:
            return self._parse_filter_details(details, result)
        elif 'Seek' in operator_type:
            return self._parse_seek_details(details, result)


    def _parse_seek_details(self, details: str, result: Dict) -> Dict:
        details = [part.strip() for part in details.split(',')]
        details = details[0]  # 主要的扫描信息
        NODE_SEEK_PATTERN = re.compile(
            r'^\s*'
            r'(?:'
                r'(UNIQUE)\s+'
                r'|'
                r'((?:RANGE|BTREE|LOOKUP))\s+INDEX\s+'
            r')?'
            r'(?P<var>\w+):(?P<label>\w+)\((?P<prop>[\w,\s]+)\)\s*'
            r'(?:WHERE\s+'
                r'(?P<lhs>\w+)\s*'
                r'(?P<op>=|<>|!=|<=|>=|<|>|STARTS\s+WITH|ENDS\s+WITH|CONTAINS)\s*'
                r'(?P<rhs>\$[\w\.]+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w\.\-]+)'
            r')?'
            r'\s*$',
            re.IGNORECASE
        )
        # 在原来基础上增加箭头方向的命名组
        REL_SEEK_PATTERN = re.compile(
            r'^\s*'
            r'(?:(?P<index_type>(?:RANGE|BTREE|LOOKUP))\s+INDEX\s+)?'
            r'\('
                r'(?P<src>\w+)'
            r'\)\s*'
            r'(?P<left_arrow><-|\-)\s*'          # 命名组：左侧箭头
            r'\['
                r'(?P<relvar>\w+):(?P<reltype>\w+)\((?P<prop>[\w,\s]+)\)'
            r'\]\s*'
            r'(?P<right_arrow>\->|\-)\s*'        # 命名组：右侧箭头
            r'\('
                r'(?P<dst>\w+)'
            r'\)\s*'
            r'(?:WHERE\s+'
                r'(?P<lhs>\w+)\s*'
                r'(?P<op>=|<>|!=|<=|>=|<|>|STARTS\s+WITH|ENDS\s+WITH|CONTAINS)\s*'
                r'(?P<rhs>\$[\w\.]+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w\.\-]+)'
            r')?'
            r'\s*$',
            re.IGNORECASE
        )
        # 用于 MultiNodeIndexSeek 的模式
        MULTI_NODE_SEEK_PATTERN = re.compile(
            r'(?:'
                r'(?:(UNIQUE)\s+|((?:RANGE|BTREE|LOOKUP))\s+INDEX\s+)?'
                r'(?P<var>\w+):(?P<label>\w+)\((?P<prop>[\w,\s]+)\)\s*'
                r'(?:WHERE\s+'
                    r'(?P<lhs>\w+)\s*'
                    r'(?P<op>=|<>|!=|<=|>=|<|>|STARTS\s+WITH|ENDS\s+WITH|CONTAINS)\s*'
                    r'(?P<rhs>\$[\w\.]+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w\.\-]+)'
                r')?'
            r')',
            re.IGNORECASE
        )
        if ',' in details and 'INDEX' in details:
            # 找到所有匹配
            matches = MULTI_NODE_SEEK_PATTERN.findall(details)
            for i, match in enumerate(matches):
                variable = match[2]         # location
                node_label = match[3]       # Location
                properties_str = match[4]   # name
                # lhs = match[5] if len(match) > 5 else None      # name (WHERE)
                # operator = match[6] if len(match) > 6 else None # =
                # rhs = match[7] if len(match) > 7 else None      # $autostring_0
                # 添加节点标签
                result['nodes'].add(node_label)
                # 处理属性
                properties = [prop.strip() for prop in properties_str.split(',')]
                prop_set = set(properties)
                for prop in properties:
                    result['properties'].add(prop)
                # 添加WHERE属性
                # if lhs and lhs not in prop_set:
                #     result['properties'].add(lhs)
                #     prop_set.add(lhs)
                # 设置节点属性映射
                if node_label in result['node_properties']:
                    result['node_properties'][node_label].update(prop_set)
                else:
                    result['node_properties'][node_label] = prop_set
            return result
        node_match = NODE_SEEK_PATTERN.match(details)
        if node_match:
            # 提取匹配组
            variable = node_match.group('var')       # location
            node_label = node_match.group('label')   # Location
            properties_str = node_match.group('prop') # name 或 name,age
            # lhs = node_match.group('lhs')            # name (WHERE左边)
            # operator = node_match.group('op')        # = 或 STARTS WITH 等
            # rhs = node_match.group('rhs')            # $autostring_0
            # 添加节点标签
            result['nodes'].add((variable, node_label))
            # 处理属性（可能是复合属性）
            properties = [prop.strip() for prop in properties_str.split(',')]
            prop_set = set(properties)
            # 添加属性到结果
            for prop in properties:
                result['properties'].add(prop)
            # 添加WHERE属性（如果不同）
            # if lhs and lhs not in prop_set:
            #     result['properties'].add(lhs)
            #     prop_set.add(lhs)
            # 设置节点属性映射
            if node_label in result['node_properties']:
                result['node_properties'][node_label].update(prop_set)
            else:
                result['node_properties'][node_label] = prop_set
            return result
        rel_match = REL_SEEK_PATTERN.match(details)
        if rel_match:
            # 一次性提取所有信息
            # index_type = rel_match.group('index_type')      # RANGE/BTREE/LOOKUP
            source_var = rel_match.group('src')             # candidate
            left_arrow = rel_match.group('left_arrow')      # <-
            rel_var = rel_match.group('relvar')             # r
            rel_type = rel_match.group('reltype')           # WORKS_IN
            properties_str = rel_match.group('prop')        # title
            right_arrow = rel_match.group('right_arrow')    # -
            target_var = rel_match.group('dst')             # anon_0
            # lhs = rel_match.group('lhs')                    # title (WHERE)
            # operator = rel_match.group('op')                # =
            # rhs = rel_match.group('rhs')                    # $autostring_0
            if (left_arrow == '<-' and right_arrow == '-'):
                temp = source_var
                source_var = target_var
                target_var = temp
            # 添加关系（参考 expand 的格式）
            result['relationships'].add((rel_var, rel_type))
            # 处理属性（已经从正则中提取）
            properties = [prop.strip() for prop in properties_str.split(',')]
            prop_set = set(properties)
            for prop in properties:
                result['properties'].add(prop)
            # if lhs and lhs not in prop_set:
            #     result['properties'].add(lhs)
            #     prop_set.add(lhs)
            if rel_type in result['rel_properties']:
                result['rel_properties'][rel_type].update(prop_set)
            else:
                result['rel_properties'][rel_type] = prop_set
            # 添加图结构（参考 expand 的格式）
            result['graph_structure'].append((source_var, rel_var, target_var))
            return result
        return result
    
    
    def _parse_scan_details(self, details: str, result: Dict) -> Dict:
        # """解析扫描操作：u:University"""
        # matches = self.label_pattern.findall(details)
        # for var, label in matches:
        #     result['nodes'].add((var,label))
        # return result
        # 1. 节点标签扫描：person:Person
        details = [part.strip() for part in details.split(',')]
        details = details[0]  # 主要的扫描信息
        NODE_LABEL_PATTERN = re.compile(
            r'^\s*(?P<var>\w+):(?P<label>\w+)\s*$',
            re.IGNORECASE
        )
        # 2. 交集节点扫描：countryAndLocation:Country&Location
        INTERSECTION_PATTERN = re.compile(
            r'^\s*(?P<var>\w+):(?P<labels>\w+(?:&\w+)+)\s*$',
            re.IGNORECASE
        )
        # 3. 差集节点扫描：locationNotCountry:Location&!Country
        SUBTRACTION_PATTERN = re.compile(
            r'^\s*(?P<var>\w+):(?P<labels>\w+(?:&!\w+)+)\s*$',
            re.IGNORECASE
        )
        # 4. 并集节点扫描：countryOrLocation:Country|Location
        UNION_PATTERN = re.compile(
            r'^\s*(?P<var>\w+):(?P<labels>\w+(?:\|\w+)+)\s*$',
            re.IGNORECASE
        )
        # 5. 节点索引扫描：TEXT INDEX l:Location(name) WHERE name CONTAINS $autostring_0
        NODE_INDEX_SCAN_PATTERN = re.compile(
            r'^\s*'
            r'(?P<index_type>TEXT|RANGE|BTREE|LOOKUP)\s+INDEX\s+'
            r'(?P<var>\w+):(?P<label>\w+)\((?P<prop>[\w,\s]+)\)\s*'
            r'WHERE\s+'
            r'(?P<lhs>\w+)\s*'
            r'(?P<op>CONTAINS|ENDS\s+WITH|STARTS\s+WITH|IS\s+NOT\s+NULL|IS\s+NULL|=|<>|!=|<=|>=|<|>)\s*'
            r'(?P<rhs>\$[\w\.]+|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[\w\.\-]+|NULL)?'
            r'\s*$',
            re.IGNORECASE
        )
        # 6. 关系扫描模式（参考 REL_SEEK_PATTERN）
        # 无向全关系扫描：(anon_0)-[r]-(anon_1)
        # 有向关系类型扫描：(anon_0)-[r:FRIENDS_WITH]->(anon_1)
        REL_SCAN_PATTERN = re.compile(
            r'^\s*'
            r'\('
                r'(?P<src>\w+)'
            r'\)\s*'
            r'(?P<left_arrow><-|\-)\s*'          # 左侧箭头
            r'\['
                r'(?P<relvar>\w+)(?::(?P<reltype>\w+))?'  # 关系变量，可选关系类型
            r'\]\s*'
            r'(?P<right_arrow>\->|\-)\s*'        # 右侧箭头
            r'\('
                r'(?P<dst>\w+)'
            r'\)\s*$',
            re.IGNORECASE
        )
        index_scan_match = NODE_INDEX_SCAN_PATTERN.match(details)
        if index_scan_match:
            variable = index_scan_match.group('var')             # l
            node_label = index_scan_match.group('label')         # Location
            properties_str = index_scan_match.group('prop')      # name
            lhs = index_scan_match.group('lhs')                  # name
            # operator = index_scan_match.group('op')              # CONTAINS/ENDS WITH/IS NOT NULL
            # rhs = index_scan_match.group('rhs')                  # $autostring_0 或 NULL
            # 添加节点（参考其他函数的格式）
            result['nodes'].add((variable, node_label))
            # 处理属性
            properties = [prop.strip() for prop in properties_str.split(',')]
            prop_set = set(properties)
            for prop in properties:
                result['properties'].add(prop)
            # 添加WHERE条件中的属性
            # if lhs and lhs not in prop_set:
            #     result['properties'].add(lhs)
            #     prop_set.add(lhs)
            # 设置节点属性映射
            if node_label in result['node_properties']:
                result['node_properties'][node_label].update(prop_set)
            else:
                result['node_properties'][node_label] = prop_set
            return result
        
        rel_scan_match = REL_SCAN_PATTERN.match(details)
        if rel_scan_match:
            source_var = rel_scan_match.group('src')             # anon_0
            left_arrow = rel_scan_match.group('left_arrow')      # <- 或 -
            rel_var = rel_scan_match.group('relvar')             # r
            rel_type = rel_scan_match.group('reltype')           # FRIENDS_WITH 或 None
            right_arrow = rel_scan_match.group('right_arrow')    # -> 或 -
            target_var = rel_scan_match.group('dst')             # anon_1
            if (left_arrow == '<-' and right_arrow == '-'):
                temp = source_var
                source_var = target_var
                target_var = temp
            if rel_type:
                result['relationships'].add((rel_var, rel_type))
            else:
                result['relationships'].add((rel_var, '*'))  # * 表示所有关系类型
            result['graph_structure'].append((source_var, rel_var, target_var))
            return result
        
        intersection_match = INTERSECTION_PATTERN.match(details)
        if intersection_match:
            variable = intersection_match.group('var')           # countryAndLocation
            labels_str = intersection_match.group('labels')      # Country&Location
            labels = [label.strip() for label in labels_str.split('&')]
            for label in labels:
                result['nodes'].add((variable, label))
            return result
        
        subtraction_match = SUBTRACTION_PATTERN.match(details)
        if subtraction_match:
            variable = subtraction_match.group('var')            # locationNotCountry
            labels_str = subtraction_match.group('labels')       # Location&!Country
            # 解析标签，处理 ! 前缀
            include_labels = []
            exclude_labels = []
            for part in labels_str.split('&'):
                part = part.strip()
                if part.startswith('!'):
                    exclude_labels.append(part[1:])  # 去掉 ! 前缀
                else:
                    include_labels.append(part)
            # 添加包含的标签
            for label in include_labels:
                result['nodes'].add((variable, label))
            return result
        
        union_match = UNION_PATTERN.match(details)
        if union_match:
            variable = union_match.group('var')                  # countryOrLocation
            labels_str = union_match.group('labels')             # Country|Location
            # 解析标签
            labels = [label.strip() for label in labels_str.split('|')]
            for label in labels:
                result['nodes'].add((variable, label))
            return result
        
        node_label_match = NODE_LABEL_PATTERN.match(details)
        if node_label_match:
            variable = node_label_match.group('var')             # person
            node_label = node_label_match.group('label')         # Person
            # 添加节点
            result['nodes'].add((variable, node_label))
            return result
        return result


    def _parse_expand_details(self, details: str, result: Dict) -> Dict:
        # """解析扩展操作：(u)<-[s:STUDY_AT]-(p2)"""
        # matches = self.expand_pattern.findall(details)
        # for source_var, left_arr, rel_var, rel_type, right_arr, target_var in matches:
        #     if (left_arr == '<-' and right_arr == '-'):
        #         temp = source_var
        #         source_var = target_var
        #         target_var = temp
        #     # else if(left_arr == '-' and right_arr == '->'):
        #         # pass
        #     result['relationships'].add((rel_var,rel_type))
        #     # 添加图结构，这里用变量名占位，实际标签在Filter中确定
        #     result['graph_structure'].append((source_var, rel_var, target_var))
        # return result
        # 处理有问题：
        # "(p)-[anon_0:FRIENDS_WITH*..2]-(q)",
        # "(p)-[:FRIENDS_WITH*3..4]-(q)",
        # "(p)-[:FRIENDS_WITH*..4]-(q)"

        EXPAND_PATTERN = re.compile(
            r'^\s*'
            r'\('
                r'(?P<src>\w+)'
            r'\)\s*'
            r'(?P<left_arrow><-|\-)\s*'
            r'\['
                r'(?P<relvar>\w+)(?::(?P<reltype>\w+))?'  # 关系类型可选
            r'\]\s*'
            r'(?P<right_arrow>\->|\-)\s*'
            r'\('
                r'(?P<dst>\w+)'
            r'\)'
            r'(?:\s+WHERE\s+(?P<where_clause>.+?))?'     # WHERE 条件可选
            r'\s*$',
            re.IGNORECASE
        )
        match = EXPAND_PATTERN.match(details)
        if not match:
            return result
        source_var = match.group('src')         # k
        left_arrow = match.group('left_arrow')  # -
        rel_var = match.group('relvar')         # anon_1
        rel_type = match.group('reltype')       # WORK_AT
        right_arrow = match.group('right_arrow') # ->
        target_var = match.group('dst')         # c
        # 处理箭头方向
        if (left_arrow == '<-' and right_arrow == '-'):
            temp = source_var
            source_var = target_var
            target_var = temp
        # 添加节点（使用空字符串作为标签，因为 expand 中没有指定节点标签）
        result['nodes'].add((source_var, ''))   # ('k', '')
        result['nodes'].add((target_var, ''))   # ('c', '')
        # 添加关系
        result['relationships'].add((rel_var, rel_type))  # ('anon_1', 'WORK_AT')
        # 添加图结构
        result['graph_structure'].append((source_var, rel_var, target_var))  # ('k', 'anon_1', 'c')
        return result
    
    
    def _parse_filter_details(self, details: str, result: Dict) -> Dict:
        # """解析过滤条件：p.gender = $autostring_2 AND p:Person""" # TODO: fixme
        # # 解析标签过滤
        # label_matches = self.label_pattern.findall(details)
        # for var, label in label_matches:
        #     result['nodes'].add((var,label))
        # # 解析属性访问
        # prop_matches = self.property_pattern.findall(details)
        # for var, prop in prop_matches:
        #     result['properties'].add(f"{var}.{prop}")
        #     # 这里需要根据上下文确定var对应的标签
        # # 解析缓存属性
        # cache_matches = self.cache_property_pattern.findall(details)
        # for var, prop in cache_matches:
        #     result['properties'].add(f"{var}.{prop}")
        # return result
        # 1. 节点标签模式：c:Company
        NODE_LABEL_PATTERN = re.compile(r'(\w+):(\w+)', re.IGNORECASE)
        # 2. 属性访问模式：c.id, person.name 等
        PROPERTY_ACCESS_PATTERN = re.compile(r'(\w+)\.(\w+)', re.IGNORECASE)
        # 3. 关系标签模式：r:WORKS_AT
        REL_LABEL_PATTERN = re.compile(r'(\w+):(\w+)', re.IGNORECASE)
        # ========================================
        # 解析节点标签：c:Company
        # ========================================
        node_label_matches = NODE_LABEL_PATTERN.findall(details)
        for var_name, label_name in node_label_matches:
            result['nodes'].add((var_name, label_name))
        # ========================================
        # 解析属性访问：c.id, person.name 等
        # ========================================
        property_matches = PROPERTY_ACCESS_PATTERN.findall(details)
        for var_name, prop_name in property_matches:
            # 添加属性名（不包含变量前缀）
            result['properties'].add(prop_name)
            # 尝试确定这个变量对应的节点标签
            # 首先从当前解析的节点标签中查找
            var_label = None
            for node_var, node_label in result['nodes']:
                if node_var == var_name:
                    var_label = node_label
                    break
            # 如果找到了对应的节点标签，添加到 node_properties
            if var_label:
                if var_label in result['node_properties']:
                    result['node_properties'][var_label].add(prop_name)
                else:
                    result['node_properties'][var_label] = {prop_name}
        return result
    
# def test_seek_patterns():
#     parser = DetailsParser()
#     test_cases = [
#         "RANGE INDEX t:Tag(name) WHERE name IS NOT NULL, cache[t.name]",
#         # "(p)-[anon_0:FRIENDS_WITH]->(fof)", 
#         # "(p)-[anon_0:FRIENDS_WITH]->(fof)",
#         # "(p)-[works_in:WORKS_IN]->(l) WHERE works_in.duration > $autoint_0",
#         # "(l)-[anon_0]->(p)",
#         # "(p)-[anon_0:FRIENDS_WITH*..2]-(q)",
#         # "(p)-[:FRIENDS_WITH*3..4]-(q)",
#         # "(p)-[:FRIENDS_WITH*..4]-(q)"
#     ]
#     for case in test_cases:
#         print(f"\n输入: {case}")
#         result = parser._parse_scan_details(case, {
#             'nodes': set(),
#             'relationships': set(), 
#             'properties': set(),
#             'node_properties': {},
#             'rel_properties': {},
#             'graph_structure': []
#         })
#         print(f"结果: {result}")
# test_seek_patterns()