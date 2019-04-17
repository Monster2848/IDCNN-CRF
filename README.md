# 环境：
    python3.6.3
    tensorflow 1.9.0

# 结构目录
    data_path               存储数据样别等数据
    log                     程序运行日志
    output                  训练100轮预测对比与测试对比
    parameter.py            相关的配置参数
    utils.py                数据处理
    model.py                网络模型
    run.py                  开始

# 代码运行
    Parameter——>Txt——>TFrecord——>Crf_model——>run
    关系：从Parameter继承到run
    
    运行方式：
    
    从run中到Crf_model.__init__()
    如果file_generation为True将运行utils中的TFrecord与Txt进行数据的重新加载
    
    __model_main()模型构建
        __placeholder()数据的读取
        __word_embedding()数据向量化
        __idcnn()模型网络
        __loss_layer()求loss,与CRF转置矩阵
        __Optimizer()收敛网络
        
# idcnn网络
    运行方式与传统cnn原理大致一样,只是在卷积核直接填充一个位置,使cnn卷积核大小不变的情况下扫描视野更加输入矩阵上更广阔的数据信息
