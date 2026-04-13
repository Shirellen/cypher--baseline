# 论文Pipeline图

## 查询表示学习模型架构

```mermaid
graph TB
    %% 输入层
    Input[Cypher Query] --> PlanParser[Query Plan Parser]
    Input --> SchemaExtractor[Schema Extractor]
    
    %% 编码层
    PlanParser --> TreeFeatures[Plan Tree Features]
    SchemaExtractor --> SchemaGraph[Schema Graph]
    
    %% 核心模型层
    TreeFeatures --> TreeEncoder[Plan Tree Encoder<br/>Transformer]
    SchemaGraph --> GraphEncoder[Schema Graph Encoder<br/>GAT]
    
    %% 特征融合
    TreeEncoder --> Fusion[Feature Fusion]
    GraphEncoder --> Fusion
    
    %% 表示输出
    Fusion --> Embedding[Query Embedding<br/>Unified Representation]
    
    %% 下游任务
    Embedding --> CostEst[Cost Estimation<br/>Regression]
    Embedding --> CardEst[Cardinality Estimation<br/>Regression]
    Embedding --> Clustering[Query Clustering<br/>Unsupervised]
    
    %% 样式定义
    classDef inputStyle fill:#E8F4F8,stroke:#01579B,stroke-width:2px
    classDef encoderStyle fill:#B3E5FC,stroke:#01579B,stroke-width:2px
    classDef coreStyle fill:#4FC3F7,stroke:#01579B,stroke-width:3px
    classDef fusionStyle fill:#0288D1,stroke:#01579B,stroke-width:3px,color:white
    classDef embeddingStyle fill:#01579B,stroke:#01579B,stroke-width:4px,color:white
    classDef taskStyle fill:#FFA726,stroke:#E65100,stroke-width:2px
    
    class Input inputStyle
    class PlanParser,SchemaExtractor,TreeFeatures,SchemaGraph encoderStyle
    class TreeEncoder,GraphEncoder coreStyle
    class Fusion fusionStyle
    class Embedding embeddingStyle
    class CostEst,CardEst,Clustering taskStyle
```

## 简化版本（适合论文正文）

```mermaid
graph LR
    %% 输入
    Query[Query] --> Encoder[Query Representation<br/>Learning Model]
    
    %% 核心模型
    Encoder --> |Unified Embedding| Rep[Query Embedding]
    
    %% 下游任务
    Rep --> Task1[Cost Estimation]
    Rep --> Task2[Cardinality Estimation]
    Rep --> Task3[Query Clustering]
    
    %% 样式
    classDef queryStyle fill:#E8F4F8,stroke:#01579B,stroke-width:2px
    classDef encoderStyle fill:#4FC3F7,stroke:#01579B,stroke-width:3px
    classDef repStyle fill:#01579B,stroke:#01579B,stroke-width:4px,color:white
    classDef taskStyle fill:#FFA726,stroke:#E65100,stroke-width:2px
    
    class Query queryStyle
    class Encoder encoderStyle
    class Rep repStyle
    class Task1,Task2,Task3 taskStyle
```

## 详细架构图（适合技术章节）

```mermaid
graph TB
    %% 输入层
    subgraph Input[Input Layer]
        Q[Cypher Query]
    end
    
    %% 解析层
    subgraph Parsing[Parsing Layer]
        P1[Query Plan Parser]
        P2[Schema Extractor]
    end
    
    %% 特征提取层
    subgraph Features[Feature Extraction Layer]
        F1[Node Types]
        F2[Join Operations]
        F3[Filter Conditions]
        F4[Histogram Statistics]
        F5[Schema Graph]
    end
    
    %% 编码层
    subgraph Encoding[Encoding Layer]
        E1[Plan Tree Encoder<br/>Transformer]
        E2[Schema Graph Encoder<br/>GAT]
    end
    
    %% 融合层
    subgraph Fusion[Feature Fusion Layer]
        M1[Multi-head Attention]
        M2[Feature Concatenation]
        M3[Projection Layer]
    end
    
    %% 输出层
    subgraph Output[Output Layer]
        O1[Query Embedding]
    end
    
    %% 下游任务
    subgraph Tasks[Downstream Tasks]
        T1[Cost Estimation]
        T2[Cardinality Estimation]
        T3[Query Clustering]
    end
    
    %% 连接关系
    Q --> P1
    Q --> P2
    P1 --> F1
    P1 --> F2
    P1 --> F3
    P1 --> F4
    P2 --> F5
    
    F1 --> E1
    F2 --> E1
    F3 --> E1
    F4 --> E1
    F5 --> E2
    
    E1 --> M1
    E2 --> M2
    M1 --> M2
    M2 --> M3
    M3 --> O1
    
    O1 --> T1
    O1 --> T2
    O1 --> T3
    
    %% 样式
    classDef inputStyle fill:#E8F4F8,stroke:#01579B,stroke-width:2px
    classDef parseStyle fill:#B3E5FC,stroke:#01579B,stroke-width:2px
    classDef featureStyle fill:#81D4FA,stroke:#01579B,stroke-width:2px
    classDef encodeStyle fill:#4FC3F7,stroke:#01579B,stroke-width:3px
    classDef fusionStyle fill:#0288D1,stroke:#01579B,stroke-width:3px,color:white
    classDef outputStyle fill:#01579B,stroke:#01579B,stroke-width:4px,color:white
    classDef taskStyle fill:#FFA726,stroke:#E65100,stroke-width:2px
    
    class Q inputStyle
    class P1,P2 parseStyle
    class F1,F2,F3,F4,F5 featureStyle
    class E1,E2 encodeStyle
    class M1,M2,M3 fusionStyle
    class O1 outputStyle
    class T1,T2,T3 taskStyle
```

## 使用说明

1. **完整版**：适合论文首页或架构章节，展示完整的模型流程
2. **简化版**：适合摘要或引言部分，突出核心思想
3. **详细版**：适合技术章节，展示各层细节

## 导出为图片

你可以使用以下工具将Mermaid图导出为图片：
- Mermaid Live Editor: https://mermaid.live/
- VS Code插件: Markdown Preview Mermaid Support
- 在线工具: https://github.com/mermaid-js/mermaid-cli

## 论文中的建议使用方式

1. **Figure 1**: 使用简化版展示整体框架
2. **Figure 2**: 使用完整版展示模型架构
3. **Figure 3**: 使用详细版展示技术细节
