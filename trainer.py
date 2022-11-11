import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
from tqdm import tqdm
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv("./data/train.csv")  # 官网下载的训练集
submit_example = pd.read_csv("./data/submit_example.csv")  # 提交测试结果的样例
evaluation_public = pd.read_csv("./data/evaluation_public.csv")  # 测试集

# 把训练集和测试集的'is_test'分别【标记】为False和True，方便后面划分数据
train['is_test'] = False
evaluation_public['is_test'] = True

# 合并训练集和测试集，一次性把训练集和测试集数据都处理了，比如缺失值填充、编码
# pd.concat（）基于同一轴将多个数据集合并
all_data = pd.concat([train, evaluation_public]).reset_index(drop=True)
features = []
label = 'is_risk'

# # 获取时间特征
all_data['op_datetime'] = pd.to_datetime(all_data['op_datetime'])  # 数据类型从object到datetime64
all_data['day'] = all_data['op_datetime'].dt.day  # pandas.Series.dt.day 取天
all_data['weekday'] = all_data['op_datetime'].dt.weekday
all_data['month'] = all_data['op_datetime'].dt.month
all_data['hour'] = all_data['op_datetime'].dt.hour
all_data['ts'] = all_data['hour'] * 60 + all_data['op_datetime'].dt.minute  # 取分钟

features = ['weekday', 'hour', 'ts']

# 缺失值处理和字符编码，经过分析发现大部分缺失值都是类别型变量（类似于性别这种）
# 缺失的属性有user_name 用户名、department 用户所在部门、ip_type	IP类型、log_system_transform	接入系统(加密后)
# 因此直接对缺失值赋值-1

for f in all_data.columns:  # all_data.columns返回一个index类型的列索引列表
    if f not in ['id']:  # 不是id列
        all_data[f] = all_data[f].fillna(-1)  # fillna（）填充缺失值   除了id列，其他列的缺失值都用-1填充
for f in all_data.columns:
    if f not in {'id', 'op_month', 'is_risk', 'is_test', 'weekday', 'hour', 'ts'}:
        all_data[f] = all_data[f].astype('str')  # 转换为字符串
        all_data[f] = LabelEncoder().fit_transform(all_data[f]) # 对该列的值进行编码
        features.append(f)

# 提取用户特征，提取用户特征的时候切忌获取到用户未来的特征，防止特征穿越，通过统计总共有188个用户，对每个用户分别进行特征计算
# 按照日期进行降序排列
# ignore_index: 是否对行索引进行重新的排序    ignore_index=True 对行索引重新排序
all_data.sort_values(by=['op_datetime'], ascending=True, inplace=True, ignore_index=True)

# 获取去重后的用户名列表 user_list
user_list = all_data['user_name'].drop_duplicates()  # drop_duplicates() 去除完全重复的行数据

data_list = []
i = 0
for user_name in tqdm(user_list):  # tqdm（）在循环中显示进度条
    # all_data[all_data['user_name'] == user_name]
    # 提取出all_data中user_name列和user_name相同的 行

    # copy(deep=True) deep=True 表示深拷贝
    # 即tmp和all_data分别占用内存，此时tmp和all_data相互独立，对tmp的改动不会映射到all_data中
    # tmp保存当前用户的所有访问记录
    tmp = all_data[all_data['user_name'] == user_name].copy(deep=True).reset_index(drop=True)
    # 统计这些特征截止到当前的一些数据情况
    for f in ['department', 'ip_transform', 'device_num_transform',
              'browser_version', 'browser', 'os_type', 'os_version', 'op_datetime',
              'ip_type', 'http_status_code', 'op_city', 'log_system_transform', 'url']:
        # 滑动窗口函数rolling（）  窗口会一个单位一个单位地滑行  【适合移动计算】
        # 假设我们有10天的销售额，想每三天求一次总和，比如第五天的总和就是第三天 + 第四天 + 第五天的销售额之和
        # window=10 滑动窗口的大小
        # min_periods=1 每个窗口最少包含的观测值数量,默认它是和窗口的长度相等的
        # center	是否将窗口标签设置为居中,默认为False,表示从当前元素往上筛选;
        # 将center指定为True的话，那么是以当前元素为中心，从两个方向上进行筛选


        rolling_data = tmp[f].rolling(window=10, min_periods=1, center=False)


        # 里面的参数x就是【每个窗口】里面的元素组成的【Series对象】
        # 这里的x指的是当前的用户的【10】条访问记录对应的【属性列的取值】  <== window=10
        # 比如循环到op_city属性的时候， x就是这个用户的10个op_city取值

        # agg()是聚合函数
        # len（）  返回对象(字符、列表、元组等)长度或项目个数
        # lambda x: len(set(x))返回的是当前用户的对应属性的【10】个取值去重后的属性值个数，比如10个op_city取值去重后的op_city个数
        # 但因为是滑动窗口，rolling_ data的前1~9个属性值【分别】是第1~9个滑动窗口（未达到窗口大小)，分别执行第①个lambda函数
        # 再第一个10值窗口（从第10个属性值往前，共10个取值）执行第①个lambda语句
        # 再第二个10值窗口（从第11个属性值往前，共10个取值）执行第①个lambda语句
        # 一直到窗口滑到终点，就是每个滑动窗口都执行了①个lambda语句
        # 然后同理执行第②个lambda语句
        # 所以rolling_data.agg(lambda x: len(set(x)))返回的是【每个滑动窗口】包含的对应属性取值去重后的属性值个数
        # 也就是说返回的是一个series对象

        # f'{}'格式化字符串，花括号里放变量或表达式，比如tmp[op_city_unique]
        # tmp存放的是一个用户的所有的访问记录，是个dataframe
        # 而rolling_data.agg(lambda x: len(set(x)))返回的是一个series对象,
        # 那tmp[f'{f}_unique']这一列只有一个值，就是这个series对象
        tmp[f'{f}_unique'] = rolling_data.agg(lambda x: len(set(x)))  # set()去重  第①个lambda语句

        # lambda x: len(x)返回的是当前用户的对应属性的【10】个取值的属性值个数（为去重），比如10个op_city取值的op_city个数
        # rolling_data.agg(lambda x: len(x))返回的是【每个滑动窗口】包含的对应属性取值未去重的属性值个数
        # rolling_data.agg(lambda x: len(x)) 返回的是一个series对象
        tmp[f'{f}_count'] = rolling_data.agg(lambda x: len(x))  # 第②个lambda语句

        # stats.mode()求众数，如果众数的值不唯一，只显示其中一个，同时能知道该众数的出现次数数。
        # stats.mode()方法的返回值类型是 ModeResult，有两个维度，mode和count
        # 比如print(mode(a))
        # 返回ModeResult(mode=array([[3, 1, 0, 0]]), count=array([[1, 1, 1, 1]]))
        # 第0维mode=array([[3, 1, 0, 0]])表示每一列的众数，如果没有则返回最小值，第1维count=array([[1, 1, 1, 1]]表示出现的次数
        # mode(a)[0]就是array([[3, 1, 0, 0]])，即[[3, 1, 0, 0]]，mode(a)[0][0]就是[3, 1, 0, 0],mode(a)[0][0][0]就是3

        # stats.mode(x)[0][0]指的是这个滑动窗口中的众数
        # 因为x是series类型，所以stats.mode(x)返回值类似ModeResult(mode=array(['深圳'], dtype='string'), count=array([100]))
        # stats.mode(x)[0]就是array(['深圳']，即['深圳']，stats.mode(x)[0][0]就是深圳
        # rolling_data.agg(lambda x: stats.mode(x)[0][0])返回的是一个series对象
        tmp[f'{f}_mod'] = rolling_data.agg(lambda x: stats.mode(x)[0][0])  # 众数
        tmp[f'{f}_diff'] = tmp[f].diff()  # 与前一行对比 ，用该行减去前一行（对应位置的值相减），首行用null来填充
        if i == 0:
            features.append(f'{f}_unique')
            features.append(f'{f}_count')
            features.append(f'{f}_mod')
            features.append(f'{f}_diff')
    i = i + 1  # 保证f'{f}_unique'等这些特征只能加入一次到features[]
    data_list.append(tmp)  # 每一个tmp（dataframe类型） 用append()函数添加到列表当中
all_data = pd.concat(data_list).reset_index(drop=True)  # concat就是把list转换成dataframe
all_data = all_data.fillna(-1)
train = all_data[all_data['is_test'] == False].copy(deep=True)
test = all_data[all_data['is_test'] == True].copy(deep=True)

# pandas.std() 求标准差，代表样本值与平均值之间的偏离程度,默认是除以n-1
# 所有数减去平均值,它的平方和除以个数减一,再把所得值开根号,就是1/2次方,得到的数就是这组数的标准差
features = [f for f in features if train[f].std() > 0]

# 训练集和验证集的划分
# train_test_split（要划分的样本特征集，要划分的样本结果，测试集比例，随机因子）随机划分训练集和测试集
# label = 'is_risk'
# test_size=0.2 表示测试集比例
# random_state=1表示随机因子，每次测试集都一样
# X_train	划分出的训练数据集数据
# X_test	划分出的测试数据集数据
# y_train	划分出的训练数据集的标签
# y_test	划分出的测试数据集的标签
X_train, X_test, y_train, y_test = train_test_split(train[features], train[label], test_size=0.2, random_state=1)

# 模型训练，先通过训练集找一个最佳的树的棵树，在通过全量数据进行训练
model = lgb.LGBMClassifier(
    boosting="gbdt",  # 提升树的类型 gbdt（梯度提升树）,dart,goss,rf
    max_depth=4,  # 树的深度
    learning_rate=0.08,  # 学习率
    n_estimators=5000,  # 拟合的树的棵树，相当于训练轮数
    subsample=0.5,  # 每棵树用的样本比例
    subsample_freq=10,  # 子样本频率
    min_data_in_leaf=20,  # 每个叶子节点的样本数
    feature_fraction=0.8,  # 训练的时候选择的特征比例
    bagging_seed=1221,  # 表示bagging 的随机数种子
    reg_alpha=2.5,  # l1正则化因子
    reg_lambda=2.5,  # l2正则化因子
    min_sum_hessian_in_leaf=1e-1,  # 表示一个叶子节点上的最小hessian 之和
    random_state=1212  # 随机种子数
)

# eval_set 用于评估的数据集
# eval_metric 用于指定评估指标
# early_stopping_rounds指定早停轮数，即如果在150轮内验证集指标不提升我们就停止迭代
# 间隔多少次迭代输出一次信息
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=['auc'],
          early_stopping_rounds=150, verbose=100)

# predict_proba(X_test)[:, 1] 预测X_test的每条数据标签是0的概率和1的概率
# [:,1]是取二维数组中第二维的所有数据，也是说只取X_test的每条数据标签为1的概率
test_pred = model.predict_proba(X_test)[:, 1]  # 返回各个样本属于各个类别的概率

# roc_auc_score（y_true，y_score）计算曲线ROC的面积
# y_true：真实的标签   y_score：目标分数
print(f"auc = {roc_auc_score(y_true=y_test, y_score=test_pred)}")

# 模型训练
model2 = lgb.LGBMClassifier(
    boosting="gbdt",
    max_depth=4,  # 树的深度
    learning_rate=0.08,
    n_estimators=int(1.2 * model.best_iteration_),  # 拟合的树的棵数，相当于训练轮数
    subsample=0.5,  # 每棵树用的样本比例
    subsample_freq=10,  # 子样本频率
    min_data_in_leaf=20,
    feature_fraction=0.8,  # 训练的时候选择的特征比例
    bagging_seed=1221,
    reg_alpha=2.5,  # l1正则化因子
    reg_lambda=2.5,  # 此处不改了
    min_sum_hessian_in_leaf=1e-1,  # 表示一个叶子节点上的最小hessian 之和
    random_state=1212
)
model2.fit(train[features], train[label])
test['is_risk'] = model2.predict_proba(test[features])[:, 1]  # 返回各个样本属于各个类别的概率

res = test[['id', 'is_risk']].copy(deep=True)
res.sort_values(by=['id'], ascending=True, inplace=True, ignore_index=True)
res.to_csv("res.csv", index=False)
