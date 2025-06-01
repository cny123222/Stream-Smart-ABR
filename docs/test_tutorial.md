# Test流程
1. 测试结果保存在STREAM-SMART-ABR/Test目录下，可能需要手动创建一下
2. 进入STREAM-SMART-ABR目录，运行 python src/control.py
3. control.py会自动将参数传给client.py开始执行，每个测试样例执行3mins**必须保证测试的时候播放器在电脑的页面上，不能放在后台**。

# 参数说明
1. 第一个参数为int，代表选择的ABR决策，目前只有三个，用1、2、3作记号
2. 第二个参数为int，代表选择模拟的网络环境，目前只有8个记为1、2、3、4、5、6、7、8
3. 第三个参数为str，代表输出的路径，会根据参数12自动创建

# 测试结果说明
1. quality：帧率为x的视频的质量记为ln(x/360)
2. effective_ratio：视频有效播放时间占总运行时间的比值（减去了rebuffering和startup）
3. weighted_quality：quality和effective_ratio的加权和，权重分别为1.0和1.5，是随！便！取的（想法是把quality的最大值1.5和effective_ratio的最大值1.0结合起来）

# 怎么加入新的ABR决策
1. 在ABRManager类中def一个新的决策函数_cq123222nbnb(self)，完成逻辑
2. 在ABRManager类的最前面（__init__）之前加入表示该决策逻辑的枚举量
3. 在_abr_decision_logic(self)中加入一个elif
4. 与Test接口保持一致：假设这个决策函数用4作为记号，那么在client.py的main函数中加入elif abr_decision == 4的分支。
5. 在Control.py的开头更新parameter_combinations的范围

# 怎么加入新的网络环境模拟
1. 在network_simulator.py中加入新的elif分支
2. 在Control.py的开头更新parameter_combinations的范围
